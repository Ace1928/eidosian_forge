from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
class TestArrayLikes:

    @pytest.mark.parametrize('arraylike', arraylikes())
    def test_0d_object_special_case(self, arraylike):
        arr = np.array(0.0)
        obj = arraylike(arr)
        res = np.array(obj, dtype=object)
        assert_array_equal(arr, res)
        res = np.array([obj], dtype=object)
        assert res[0] is obj

    @pytest.mark.parametrize('arraylike', arraylikes())
    @pytest.mark.parametrize('arr', [np.array(0.0), np.arange(4)])
    def test_object_assignment_special_case(self, arraylike, arr):
        obj = arraylike(arr)
        empty = np.arange(1, dtype=object)
        empty[:] = [obj]
        assert empty[0] is obj

    def test_0d_generic_special_case(self):

        class ArraySubclass(np.ndarray):

            def __float__(self):
                raise TypeError('e.g. quantities raise on this')
        arr = np.array(0.0)
        obj = arr.view(ArraySubclass)
        res = np.array(obj)
        assert_array_equal(arr, res)
        with pytest.raises(TypeError):
            np.array([obj])
        obj = memoryview(arr)
        res = np.array(obj)
        assert_array_equal(arr, res)
        with pytest.raises(ValueError):
            np.array([obj])

    def test_arraylike_classes(self):
        arr = np.array(np.int64)
        assert arr[()] is np.int64
        arr = np.array([np.int64])
        assert arr[0] is np.int64

        class ArrayLike:

            @property
            def __array_interface__(self):
                pass

            @property
            def __array_struct__(self):
                pass

            def __array__(self):
                pass
        arr = np.array(ArrayLike)
        assert arr[()] is ArrayLike
        arr = np.array([ArrayLike])
        assert arr[0] is ArrayLike

    @pytest.mark.skipif(np.dtype(np.intp).itemsize < 8, reason='Needs 64bit platform')
    def test_too_large_array_error_paths(self):
        """Test the error paths, including for memory leaks"""
        arr = np.array(0, dtype='uint8')
        arr = np.broadcast_to(arr, 2 ** 62)
        for i in range(5):
            with pytest.raises(MemoryError):
                np.array(arr)
            with pytest.raises(MemoryError):
                np.array([arr])

    @pytest.mark.parametrize('attribute', ['__array_interface__', '__array__', '__array_struct__'])
    @pytest.mark.parametrize('error', [RecursionError, MemoryError])
    def test_bad_array_like_attributes(self, attribute, error):

        class BadInterface:

            def __getattr__(self, attr):
                if attr == attribute:
                    raise error
                super().__getattr__(attr)
        with pytest.raises(error):
            np.array(BadInterface())

    @pytest.mark.parametrize('error', [RecursionError, MemoryError])
    def test_bad_array_like_bad_length(self, error):

        class BadSequence:

            def __len__(self):
                raise error

            def __getitem__(self):
                return 1
        with pytest.raises(error):
            np.array(BadSequence())