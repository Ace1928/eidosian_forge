import numpy as np
class LogFunc(TransformFunction):

    def func(self, x):
        return np.log(x)

    def inverse(self, y):
        return np.exp(y)

    def deriv(self, x):
        return 1.0 / x