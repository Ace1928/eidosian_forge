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
def _get_cse_exprs():
    args = x, y = se.symbols('x y')
    exprs = [x * x + y, y / (x * x), y * x * x + x]
    inp = [11, 13]
    ref = [121 + 13, 13 / 121, 13 * 121 + 11]
    return (args, exprs, inp, ref)