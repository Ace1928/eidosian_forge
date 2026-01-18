import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
class TestMaskedObjectArray:

    def test_getitem(self):
        arr = np.ma.array([None, None])
        for dt in [float, object]:
            a0 = np.eye(2).astype(dt)
            a1 = np.eye(3).astype(dt)
            arr[0] = a0
            arr[1] = a1
            assert_(arr[0] is a0)
            assert_(arr[1] is a1)
            assert_(isinstance(arr[0, ...], MaskedArray))
            assert_(isinstance(arr[1, ...], MaskedArray))
            assert_(arr[0, ...][()] is a0)
            assert_(arr[1, ...][()] is a1)
            arr[0] = np.ma.masked
            assert_(arr[1] is a1)
            assert_(isinstance(arr[0, ...], MaskedArray))
            assert_(isinstance(arr[1, ...], MaskedArray))
            assert_equal(arr[0, ...].mask, True)
            assert_(arr[1, ...][()] is a1)
            assert_equal(arr[0].data, a0)
            assert_equal(arr[0].mask, True)
            assert_equal(arr[0, ...][()].data, a0)
            assert_equal(arr[0, ...][()].mask, True)

    def test_nested_ma(self):
        arr = np.ma.array([None, None])
        arr[0, ...] = np.array([np.ma.masked], object)[0, ...]
        assert_(arr.data[0] is np.ma.masked)
        assert_(arr[0] is np.ma.masked)
        arr[0] = np.ma.masked
        assert_(arr[0] is np.ma.masked)