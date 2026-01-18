from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
class ArrayInterface:

    def __init__(self, a):
        self.a = a
        self.__array_interface__ = a.__array_interface__