import numpy as np
class ChainFunc(TransformFunction):

    def __init__(self, finn, fout):
        self.finn = finn
        self.fout = fout

    def func(self, x):
        return self.fout.func(self.finn.func(x))

    def inverse(self, y):
        return self.f1.inverse(self.fout.inverse(y))

    def deriv(self, x):
        z = self.finn.func(x)
        return self.fout.deriv(z) * self.finn.deriv(x)