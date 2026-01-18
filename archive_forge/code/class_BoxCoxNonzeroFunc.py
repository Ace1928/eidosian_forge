import numpy as np
class BoxCoxNonzeroFunc(TransformFunction):

    def __init__(self, lamda):
        self.lamda = lamda

    def func(self, x):
        return (np.power(x, self.lamda) - 1) / self.lamda

    def inverse(self, y):
        return (self.lamda * y + 1) / self.lamda

    def deriv(self, x):
        return np.power(x, self.lamda - 1)