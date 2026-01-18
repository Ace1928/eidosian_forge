from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
class TestArrayFinalize:
    """ Tests __array_finalize__ """

    def test_receives_base(self):

        class SavesBase(np.ndarray):

            def __array_finalize__(self, obj):
                self.saved_base = self.base
        a = np.array(1).view(SavesBase)
        assert_(a.saved_base is a.base)

    def test_bad_finalize1(self):

        class BadAttributeArray(np.ndarray):

            @property
            def __array_finalize__(self):
                raise RuntimeError('boohoo!')
        with pytest.raises(TypeError, match='not callable'):
            np.arange(10).view(BadAttributeArray)

    def test_bad_finalize2(self):

        class BadAttributeArray(np.ndarray):

            def __array_finalize__(self):
                raise RuntimeError('boohoo!')
        with pytest.raises(TypeError, match='takes 1 positional'):
            np.arange(10).view(BadAttributeArray)

    def test_bad_finalize3(self):

        class BadAttributeArray(np.ndarray):

            def __array_finalize__(self, obj):
                raise RuntimeError('boohoo!')
        with pytest.raises(RuntimeError, match='boohoo!'):
            np.arange(10).view(BadAttributeArray)

    def test_lifetime_on_error(self):

        class RaisesInFinalize(np.ndarray):

            def __array_finalize__(self, obj):
                raise Exception(self)

        class Dummy:
            pass
        obj_arr = np.array(Dummy())
        obj_ref = weakref.ref(obj_arr[()])
        with assert_raises(Exception) as e:
            obj_arr.view(RaisesInFinalize)
        obj_subarray = e.exception.args[0]
        del e
        assert_(isinstance(obj_subarray, RaisesInFinalize))
        break_cycles()
        assert_(obj_ref() is not None, 'object should not already be dead')
        del obj_arr
        break_cycles()
        assert_(obj_ref() is not None, 'obj_arr should not hold the last reference')
        del obj_subarray
        break_cycles()
        assert_(obj_ref() is None, 'no references should remain')

    def test_can_use_super(self):

        class SuperFinalize(np.ndarray):

            def __array_finalize__(self, obj):
                self.saved_result = super().__array_finalize__(obj)
        a = np.array(1).view(SuperFinalize)
        assert_(a.saved_result is None)