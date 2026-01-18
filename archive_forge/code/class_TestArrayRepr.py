import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
class TestArrayRepr:

    def test_nan_inf(self):
        x = np.array([np.nan, np.inf])
        assert_equal(repr(x), 'array([nan, inf])')

    def test_subclass(self):

        class sub(np.ndarray):
            pass
        x1d = np.array([1, 2]).view(sub)
        assert_equal(repr(x1d), 'sub([1, 2])')
        x2d = np.array([[1, 2], [3, 4]]).view(sub)
        assert_equal(repr(x2d), 'sub([[1, 2],\n     [3, 4]])')
        xstruct = np.ones((2, 2), dtype=[('a', '<i4')]).view(sub)
        assert_equal(repr(xstruct), "sub([[(1,), (1,)],\n     [(1,), (1,)]], dtype=[('a', '<i4')])")

    @pytest.mark.xfail(reason='See gh-10544')
    def test_object_subclass(self):

        class sub(np.ndarray):

            def __new__(cls, inp):
                obj = np.asarray(inp).view(cls)
                return obj

            def __getitem__(self, ind):
                ret = super().__getitem__(ind)
                return sub(ret)
        x = sub([None, None])
        assert_equal(repr(x), 'sub([None, None], dtype=object)')
        assert_equal(str(x), '[None None]')
        x = sub([None, sub([None, None])])
        assert_equal(repr(x), 'sub([None, sub([None, None], dtype=object)], dtype=object)')
        assert_equal(str(x), '[None sub([None, None], dtype=object)]')

    def test_0d_object_subclass(self):

        class sub(np.ndarray):

            def __new__(cls, inp):
                obj = np.asarray(inp).view(cls)
                return obj

            def __getitem__(self, ind):
                ret = super().__getitem__(ind)
                return sub(ret)
        x = sub(1)
        assert_equal(repr(x), 'sub(1)')
        assert_equal(str(x), '1')
        x = sub([1, 1])
        assert_equal(repr(x), 'sub([1, 1])')
        assert_equal(str(x), '[1 1]')
        x = sub(None)
        assert_equal(repr(x), 'sub(None, dtype=object)')
        assert_equal(str(x), 'None')
        y = sub(None)
        x[()] = y
        y[()] = x
        assert_equal(repr(x), 'sub(sub(sub(..., dtype=object), dtype=object), dtype=object)')
        assert_equal(str(x), '...')
        x[()] = 0
        x = sub(None)
        x[()] = sub(None)
        assert_equal(repr(x), 'sub(sub(None, dtype=object), dtype=object)')
        assert_equal(str(x), 'None')

        class DuckCounter(np.ndarray):

            def __getitem__(self, item):
                result = super().__getitem__(item)
                if not isinstance(result, DuckCounter):
                    result = result[...].view(DuckCounter)
                return result

            def to_string(self):
                return {0: 'zero', 1: 'one', 2: 'two'}.get(self.item(), 'many')

            def __str__(self):
                if self.shape == ():
                    return self.to_string()
                else:
                    fmt = {'all': lambda x: x.to_string()}
                    return np.array2string(self, formatter=fmt)
        dc = np.arange(5).view(DuckCounter)
        assert_equal(str(dc), '[zero one two many many]')
        assert_equal(str(dc[0]), 'zero')

    def test_self_containing(self):
        arr0d = np.array(None)
        arr0d[()] = arr0d
        assert_equal(repr(arr0d), 'array(array(..., dtype=object), dtype=object)')
        arr0d[()] = 0
        arr1d = np.array([None, None])
        arr1d[1] = arr1d
        assert_equal(repr(arr1d), 'array([None, array(..., dtype=object)], dtype=object)')
        arr1d[1] = 0
        first = np.array(None)
        second = np.array(None)
        first[()] = second
        second[()] = first
        assert_equal(repr(first), 'array(array(array(..., dtype=object), dtype=object), dtype=object)')
        first[()] = 0

    def test_containing_list(self):
        arr1d = np.array([None, None])
        arr1d[0] = [1, 2]
        arr1d[1] = [3]
        assert_equal(repr(arr1d), 'array([list([1, 2]), list([3])], dtype=object)')

    def test_void_scalar_recursion(self):
        repr(np.void(b'test'))

    def test_fieldless_structured(self):
        no_fields = np.dtype([])
        arr_no_fields = np.empty(4, dtype=no_fields)
        assert_equal(repr(arr_no_fields), 'array([(), (), (), ()], dtype=[])')