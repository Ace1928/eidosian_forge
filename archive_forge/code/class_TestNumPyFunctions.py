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
class TestNumPyFunctions:

    def test_set_module(self):
        assert_equal(np.sum.__module__, 'numpy')
        assert_equal(np.char.equal.__module__, 'numpy.char')
        assert_equal(np.fft.fft.__module__, 'numpy.fft')
        assert_equal(np.linalg.solve.__module__, 'numpy.linalg')

    def test_inspect_sum(self):
        signature = inspect.signature(np.sum)
        assert_('axis' in signature.parameters)

    def test_override_sum(self):
        MyArray, implements = _new_duck_type_and_implements()

        @implements(np.sum)
        def _(array):
            return 'yes'
        assert_equal(np.sum(MyArray()), 'yes')

    def test_sum_on_mock_array(self):

        class ArrayProxy:

            def __init__(self, value):
                self.value = value

            def __array_function__(self, *args, **kwargs):
                return self.value.__array_function__(*args, **kwargs)

            def __array__(self, *args, **kwargs):
                return self.value.__array__(*args, **kwargs)
        proxy = ArrayProxy(mock.Mock(spec=ArrayProxy))
        proxy.value.__array_function__.return_value = 1
        result = np.sum(proxy)
        assert_equal(result, 1)
        proxy.value.__array_function__.assert_called_once_with(np.sum, (ArrayProxy,), (proxy,), {})
        proxy.value.__array__.assert_not_called()

    def test_sum_forwarding_implementation(self):

        class MyArray(np.ndarray):

            def sum(self, axis, out):
                return 'summed'

            def __array_function__(self, func, types, args, kwargs):
                return super().__array_function__(func, types, args, kwargs)
        array = np.array(1).view(MyArray)
        assert_equal(np.sum(array), 'summed')