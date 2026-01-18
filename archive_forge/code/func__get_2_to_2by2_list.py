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
def _get_2_to_2by2_list(real=True):
    args = x, y = se.symbols('x y')
    exprs = [[x + y * y, y * y], [x * y * y, se.sqrt(x) + y * y]]
    L = se.Lambdify(args, exprs, real=real)

    def check(A, inp):
        X, Y = inp
        assert A.shape[-2:] == (2, 2)
        ref = [X + Y * Y, Y * Y, X * Y * Y, cmath.sqrt(X) + Y * Y]
        ravA = ravelled(A)
        size = _size(ravA)
        for i in range(size // 4):
            for j in range(4):
                assert isclose(ravA[i * 4 + j], ref[j])
    return (L, check)