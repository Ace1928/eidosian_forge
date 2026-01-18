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
class TestArrayCreationCopyArgument(object):

    class RaiseOnBool:

        def __bool__(self):
            raise ValueError
    true_vals = [True, np._CopyMode.ALWAYS, np.True_]
    false_vals = [False, np._CopyMode.IF_NEEDED, np.False_]

    def test_scalars(self):
        for dtype in np.typecodes['All']:
            arr = np.zeros((), dtype=dtype)
            scalar = arr[()]
            pyscalar = arr.item(0)
            assert_raises(ValueError, np.array, scalar, copy=np._CopyMode.NEVER)
            assert_raises(ValueError, np.array, pyscalar, copy=np._CopyMode.NEVER)
            assert_raises(ValueError, np.array, pyscalar, copy=self.RaiseOnBool())
            assert_raises(ValueError, _multiarray_tests.npy_ensurenocopy, [1])
            with pytest.raises(ValueError):
                np.array(pyscalar, dtype=np.int64, copy=np._CopyMode.NEVER)

    def test_compatible_cast(self):

        def int_types(byteswap=False):
            int_types = np.typecodes['Integer'] + np.typecodes['UnsignedInteger']
            for int_type in int_types:
                yield np.dtype(int_type)
                if byteswap:
                    yield np.dtype(int_type).newbyteorder()
        for int1 in int_types():
            for int2 in int_types(True):
                arr = np.arange(10, dtype=int1)
                for copy in self.true_vals:
                    res = np.array(arr, copy=copy, dtype=int2)
                    assert res is not arr and res.flags.owndata
                    assert_array_equal(res, arr)
                if int1 == int2:
                    for copy in self.false_vals:
                        res = np.array(arr, copy=copy, dtype=int2)
                        assert res is arr or res.base is arr
                    res = np.array(arr, copy=np._CopyMode.NEVER, dtype=int2)
                    assert res is arr or res.base is arr
                else:
                    for copy in self.false_vals:
                        res = np.array(arr, copy=copy, dtype=int2)
                        assert res is not arr and res.flags.owndata
                        assert_array_equal(res, arr)
                    assert_raises(ValueError, np.array, arr, copy=np._CopyMode.NEVER, dtype=int2)
                    assert_raises(ValueError, np.array, arr, copy=None, dtype=int2)

    def test_buffer_interface(self):
        arr = np.arange(10)
        view = memoryview(arr)
        for copy in self.true_vals:
            res = np.array(view, copy=copy)
            assert not np.may_share_memory(arr, res)
        for copy in self.false_vals:
            res = np.array(view, copy=copy)
            assert np.may_share_memory(arr, res)
        res = np.array(view, copy=np._CopyMode.NEVER)
        assert np.may_share_memory(arr, res)

    def test_array_interfaces(self):
        base_arr = np.arange(10)

        class ArrayLike:
            __array_interface__ = base_arr.__array_interface__
        arr = ArrayLike()
        for copy, val in [(True, None), (np._CopyMode.ALWAYS, None), (False, arr), (np._CopyMode.IF_NEEDED, arr), (np._CopyMode.NEVER, arr)]:
            res = np.array(arr, copy=copy)
            assert res.base is val

    def test___array__(self):
        base_arr = np.arange(10)

        class ArrayLike:

            def __array__(self):
                return base_arr
        arr = ArrayLike()
        for copy in self.true_vals:
            res = np.array(arr, copy=copy)
            assert_array_equal(res, base_arr)
            assert res is not base_arr
        for copy in self.false_vals:
            res = np.array(arr, copy=False)
            assert_array_equal(res, base_arr)
            assert res is base_arr
        with pytest.raises(ValueError):
            np.array(arr, copy=np._CopyMode.NEVER)

    @pytest.mark.parametrize('arr', [np.ones(()), np.arange(81).reshape((9, 9))])
    @pytest.mark.parametrize('order1', ['C', 'F', None])
    @pytest.mark.parametrize('order2', ['C', 'F', 'A', 'K'])
    def test_order_mismatch(self, arr, order1, order2):
        arr = arr.copy(order1)
        if order1 == 'C':
            assert arr.flags.c_contiguous
        elif order1 == 'F':
            assert arr.flags.f_contiguous
        elif arr.ndim != 0:
            arr = arr[::2, ::2]
            assert not arr.flags.forc
        if order2 == 'C':
            no_copy_necessary = arr.flags.c_contiguous
        elif order2 == 'F':
            no_copy_necessary = arr.flags.f_contiguous
        else:
            no_copy_necessary = True
        for view in [arr, memoryview(arr)]:
            for copy in self.true_vals:
                res = np.array(view, copy=copy, order=order2)
                assert res is not arr and res.flags.owndata
                assert_array_equal(arr, res)
            if no_copy_necessary:
                for copy in self.false_vals:
                    res = np.array(view, copy=copy, order=order2)
                    if not IS_PYPY:
                        assert res is arr or res.base.obj is arr
                res = np.array(view, copy=np._CopyMode.NEVER, order=order2)
                if not IS_PYPY:
                    assert res is arr or res.base.obj is arr
            else:
                for copy in self.false_vals:
                    res = np.array(arr, copy=copy, order=order2)
                    assert_array_equal(arr, res)
                assert_raises(ValueError, np.array, view, copy=np._CopyMode.NEVER, order=order2)
                assert_raises(ValueError, np.array, view, copy=None, order=order2)

    def test_striding_not_ok(self):
        arr = np.array([[1, 2, 4], [3, 4, 5]])
        assert_raises(ValueError, np.array, arr.T, copy=np._CopyMode.NEVER, order='C')
        assert_raises(ValueError, np.array, arr.T, copy=np._CopyMode.NEVER, order='C', dtype=np.int64)
        assert_raises(ValueError, np.array, arr, copy=np._CopyMode.NEVER, order='F')
        assert_raises(ValueError, np.array, arr, copy=np._CopyMode.NEVER, order='F', dtype=np.int64)