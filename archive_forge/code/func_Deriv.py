import math
def Deriv(self, x):
    val = self.Eval(x)
    return self.beta * (1 - val * val)