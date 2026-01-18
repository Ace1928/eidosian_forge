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
def _Lambdify_heterogeneous_output(Lambdify):
    x, y = se.symbols('x, y')
    args = se.DenseMatrix(2, 1, [x, y])
    v = se.DenseMatrix(2, 1, [x ** 3 * y, (x + 1) * (y + 1)])
    jac = v.jacobian(args)
    exprs = [jac, x + y, v, (x + 1) * (y + 1)]
    lmb = Lambdify(args, *exprs)
    inp0 = (7, 11)
    inp1 = (8, 13)
    inp2 = (5, 9)
    inp = np.array([inp0, inp1, inp2])
    o_j, o_xpy, o_v, o_xty = lmb(inp)
    for idx, (X, Y) in enumerate([inp0, inp1, inp2]):
        assert np.allclose(o_j[idx, ...], [[3 * X ** 2 * Y, X ** 3], [Y + 1, X + 1]])
        assert np.allclose(o_xpy[idx, ...], [X + Y])
        assert np.allclose(o_v[idx, ...], [[X ** 3 * Y], [(X + 1) * (Y + 1)]])
        assert np.allclose(o_xty[idx, ...], [(X + 1) * (Y + 1)])