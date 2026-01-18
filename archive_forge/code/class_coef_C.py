from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from time import time
class coef_C(sympy.Function):
    """Calculate coefficients C_m for integer m.

        C_m is the coefficient of v^(2*m) in the Taylor expansion in v=0 of
        Gamma(m+1/2)/(2*pi) * (2/(rho+1))^(m+1/2) * (1-v)^(-b)
            * g(rho, v)^(-m-1/2)
        """
    nargs = 3

    @classmethod
    def eval(cls, m, rho, beta):
        if not m >= 0:
            raise ValueError('must have m >= 0')
        v = symbols('v')
        expression = (1 - v) ** (-beta) * g(2 * m, rho, v) ** (-m - Rational(1, 2))
        res = expression.diff(v, 2 * m).subs(v, 0) / factorial(2 * m)
        res = res * (gamma(m + Rational(1, 2)) / (2 * pi) * (2 / (rho + 1)) ** (m + Rational(1, 2)))
        return res