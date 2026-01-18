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
def _get_Ndim_args_exprs_funcs(order):
    args = x, y = se.symbols('x y')

    def f_a(index, _x, _y):
        a, b, c, d = index
        return _x ** a + _y ** b + (_x + _y) ** (-d)
    nd_exprs_a = np.zeros((3, 5, 1, 4), dtype=object, order=order)
    for index in np.ndindex(*nd_exprs_a.shape):
        nd_exprs_a[index] = f_a(index, x, y)

    def f_b(index, _x, _y):
        a, b, c = index
        return b / (_x + _y)
    nd_exprs_b = np.zeros((1, 7, 1), dtype=object, order=order)
    for index in np.ndindex(*nd_exprs_b.shape):
        nd_exprs_b[index] = f_b(index, x, y)
    return (args, nd_exprs_a, nd_exprs_b, f_a, f_b)