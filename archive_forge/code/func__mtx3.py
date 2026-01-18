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
def _mtx3(_x, _y):
    return [[_x ** row_idx + _y ** col_idx for col_idx in range(3)] for row_idx in range(4)]