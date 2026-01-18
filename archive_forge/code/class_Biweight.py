from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
class Biweight(CustomKernel):

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.9375 * (1 - x * x) ** 2, h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = 5.0 / 7.0
        self._kernel_var = 1.0 / 7
        self._order = 2

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.

        Special implementation optimized for Biweight.
        """
        xs, ys = self.in_domain(xs, ys, x)
        if len(xs) > 0:
            w = np.sum(square(subtract(1, square(divide(subtract(xs, x), self.h)))))
            v = np.sum(multiply(ys, square(subtract(1, square(divide(subtract(xs, x), self.h))))))
            return v / w
        else:
            return np.nan

    def smoothvar(self, xs, ys, x):
        """
        Returns the kernel smoothing estimate of the variance at point x.
        """
        xs, ys = self.in_domain(xs, ys, x)
        if len(xs) > 0:
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            rs = square(subtract(ys, fittedvals))
            w = np.sum(square(subtract(1.0, square(divide(subtract(xs, x), self.h)))))
            v = np.sum(multiply(rs, square(subtract(1, square(divide(subtract(xs, x), self.h))))))
            return v / w
        else:
            return np.nan

    def smoothconf_(self, xs, ys, x):
        """Returns the kernel smoothing estimate with confidence 1sigma bounds
        """
        xs, ys = self.in_domain(xs, ys, x)
        if len(xs) > 0:
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            rs = square(subtract(ys, fittedvals))
            w = np.sum(square(subtract(1.0, square(divide(subtract(xs, x), self.h)))))
            v = np.sum(multiply(rs, square(subtract(1, square(divide(subtract(xs, x), self.h))))))
            var = v / w
            sd = np.sqrt(var)
            K = self.L2Norm
            yhat = self.smooth(xs, ys, x)
            err = sd * K / np.sqrt(0.9375 * w * self.h)
            return (yhat - err, yhat, yhat + err)
        else:
            return (np.nan, np.nan, np.nan)