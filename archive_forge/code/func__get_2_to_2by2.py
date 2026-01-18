import array
import cmath
from functools import reduce
import itertools
from operator import mul
import math
import symengine as se
from symengine.test_utilities import raises
from symengine import have_numpy
import unittest
from unittest.case import SkipTest
def _get_2_to_2by2():
    args = x, y = se.symbols('x y')
    exprs = np.array([[x + y + 1.0, x * y], [x / y, x ** y]])
    L = se.Lambdify(args, exprs)

    def check(A, inp):
        X, Y = inp
        assert abs(A[0, 0] - (X + Y + 1.0)) < 1e-15
        assert abs(A[0, 1] - X * Y) < 1e-15
        assert abs(A[1, 0] - X / Y) < 1e-15
        assert abs(A[1, 1] - X ** Y) < 1e-13
    return (L, check)