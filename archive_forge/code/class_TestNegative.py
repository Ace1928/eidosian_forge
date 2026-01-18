import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
class TestNegative:

    def test_exceptions(self):
        a = np.ones((), dtype=np.bool_)[()]
        assert_raises(TypeError, operator.neg, a)

    def test_result(self):
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            for dt in types:
                a = np.ones((), dtype=dt)[()]
                if dt in np.typecodes['UnsignedInteger']:
                    st = np.dtype(dt).type
                    max = st(np.iinfo(dt).max)
                    assert_equal(operator.neg(a), max)
                else:
                    assert_equal(operator.neg(a) + a, 0)