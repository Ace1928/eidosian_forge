import numpy as np
class AffineFunc(TransformFunction):

    def __init__(self, constant, slope):
        self.constant = constant
        self.slope = slope

    def func(self, x):
        return self.constant + self.slope * x

    def inverse(self, y):
        return (y - self.constant) / self.slope

    def deriv(self, x):
        return self.slope