import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
class TestGetImplementingArgs:

    def test_ndarray(self):
        array = np.array(1)
        args = _get_implementing_args([array])
        assert_equal(list(args), [array])
        args = _get_implementing_args([array, array])
        assert_equal(list(args), [array])
        args = _get_implementing_args([array, 1])
        assert_equal(list(args), [array])
        args = _get_implementing_args([1, array])
        assert_equal(list(args), [array])

    def test_ndarray_subclasses(self):

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        class NoOverrideSub(np.ndarray):
            pass
        array = np.array(1).view(np.ndarray)
        override_sub = np.array(1).view(OverrideSub)
        no_override_sub = np.array(1).view(NoOverrideSub)
        args = _get_implementing_args([array, override_sub])
        assert_equal(list(args), [override_sub, array])
        args = _get_implementing_args([array, no_override_sub])
        assert_equal(list(args), [no_override_sub, array])
        args = _get_implementing_args([override_sub, no_override_sub])
        assert_equal(list(args), [override_sub, no_override_sub])

    def test_ndarray_and_duck_array(self):

        class Other:
            __array_function__ = _return_not_implemented
        array = np.array(1)
        other = Other()
        args = _get_implementing_args([other, array])
        assert_equal(list(args), [other, array])
        args = _get_implementing_args([array, other])
        assert_equal(list(args), [array, other])

    def test_ndarray_subclass_and_duck_array(self):

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        class Other:
            __array_function__ = _return_not_implemented
        array = np.array(1)
        subarray = np.array(1).view(OverrideSub)
        other = Other()
        assert_equal(_get_implementing_args([array, subarray, other]), [subarray, array, other])
        assert_equal(_get_implementing_args([array, other, subarray]), [subarray, array, other])

    def test_many_duck_arrays(self):

        class A:
            __array_function__ = _return_not_implemented

        class B(A):
            __array_function__ = _return_not_implemented

        class C(A):
            __array_function__ = _return_not_implemented

        class D:
            __array_function__ = _return_not_implemented
        a = A()
        b = B()
        c = C()
        d = D()
        assert_equal(_get_implementing_args([1]), [])
        assert_equal(_get_implementing_args([a]), [a])
        assert_equal(_get_implementing_args([a, 1]), [a])
        assert_equal(_get_implementing_args([a, a, a]), [a])
        assert_equal(_get_implementing_args([a, d, a]), [a, d])
        assert_equal(_get_implementing_args([a, b]), [b, a])
        assert_equal(_get_implementing_args([b, a]), [b, a])
        assert_equal(_get_implementing_args([a, b, c]), [b, c, a])
        assert_equal(_get_implementing_args([a, c, b]), [c, b, a])

    def test_too_many_duck_arrays(self):
        namespace = dict(__array_function__=_return_not_implemented)
        types = [type('A' + str(i), (object,), namespace) for i in range(33)]
        relevant_args = [t() for t in types]
        actual = _get_implementing_args(relevant_args[:32])
        assert_equal(actual, relevant_args[:32])
        with assert_raises_regex(TypeError, 'distinct argument types'):
            _get_implementing_args(relevant_args)