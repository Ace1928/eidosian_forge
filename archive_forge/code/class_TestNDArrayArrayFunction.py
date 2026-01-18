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
class TestNDArrayArrayFunction:

    def test_method(self):

        class Other:
            __array_function__ = _return_not_implemented

        class NoOverrideSub(np.ndarray):
            pass

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented
        array = np.array([1])
        other = Other()
        no_override_sub = array.view(NoOverrideSub)
        override_sub = array.view(OverrideSub)
        result = array.__array_function__(func=dispatched_two_arg, types=(np.ndarray,), args=(array, 1.0), kwargs={})
        assert_equal(result, 'original')
        result = array.__array_function__(func=dispatched_two_arg, types=(np.ndarray, Other), args=(array, other), kwargs={})
        assert_(result is NotImplemented)
        result = array.__array_function__(func=dispatched_two_arg, types=(np.ndarray, NoOverrideSub), args=(array, no_override_sub), kwargs={})
        assert_equal(result, 'original')
        result = array.__array_function__(func=dispatched_two_arg, types=(np.ndarray, OverrideSub), args=(array, override_sub), kwargs={})
        assert_equal(result, 'original')
        with assert_raises_regex(TypeError, 'no implementation found'):
            np.concatenate((array, other))
        expected = np.concatenate((array, array))
        result = np.concatenate((array, no_override_sub))
        assert_equal(result, expected.view(NoOverrideSub))
        result = np.concatenate((array, override_sub))
        assert_equal(result, expected.view(OverrideSub))

    def test_no_wrapper(self):
        array = np.array(1)
        func = lambda x: x
        with assert_raises_regex(AttributeError, '_implementation'):
            array.__array_function__(func=func, types=(np.ndarray,), args=(array,), kwargs={})