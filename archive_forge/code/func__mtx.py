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
def _mtx(_x, _y):
    return [[_x - _y, _y ** 2], [_x + _y, _x ** 2], [_x * _y, _x ** _y]]