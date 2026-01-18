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
def _test_Lambdify_Piecewise(Lambdify):
    x = se.symbols('x')
    p = se.Piecewise((-x, x < 0), (x * x * x, True))
    f = Lambdify([x], [p])
    arr = np.linspace(3, 7)
    assert np.allclose(f(-arr).flat, arr, atol=1e-14, rtol=1e-15)
    assert np.allclose(f(arr).flat, arr ** 3, atol=1e-14, rtol=1e-15)