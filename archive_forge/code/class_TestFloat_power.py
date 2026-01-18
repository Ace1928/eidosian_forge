import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
class TestFloat_power:

    def test_type_conversion(self):
        arg_type = '?bhilBHILefdgFDG'
        res_type = 'ddddddddddddgDDG'
        for dtin, dtout in zip(arg_type, res_type):
            msg = 'dtin: %s, dtout: %s' % (dtin, dtout)
            arg = np.ones(1, dtype=dtin)
            res = np.float_power(arg, arg)
            assert_(res.dtype.name == np.dtype(dtout).name, msg)