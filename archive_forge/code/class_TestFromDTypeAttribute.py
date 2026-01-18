import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
class TestFromDTypeAttribute:

    def test_simple(self):

        class dt:
            dtype = np.dtype('f8')
        assert np.dtype(dt) == np.float64
        assert np.dtype(dt()) == np.float64

    @pytest.mark.skipif(IS_PYSTON, reason='Pyston disables recursion checking')
    def test_recursion(self):

        class dt:
            pass
        dt.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt)
        dt_instance = dt()
        dt_instance.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt_instance)

    def test_void_subtype(self):

        class dt(np.void):
            dtype = np.dtype('f,f')
        np.dtype(dt)
        np.dtype(dt(1))

    @pytest.mark.skipif(IS_PYSTON, reason='Pyston disables recursion checking')
    def test_void_subtype_recursion(self):

        class vdt(np.void):
            pass
        vdt.dtype = vdt
        with pytest.raises(RecursionError):
            np.dtype(vdt)
        with pytest.raises(RecursionError):
            np.dtype(vdt(1))