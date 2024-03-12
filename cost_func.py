import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])
w = 200
b = 0


def compute_cost(x, y, w, b):
    m = len(x_train)

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


tmp_f_wb = compute_cost(x_train, y_train, w, b)
plt.plot(tmp_f_wb, c='b', label='Our Prediction')

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title('Housing Prices')
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 square feets)')
plt.legend()
plt.show()

