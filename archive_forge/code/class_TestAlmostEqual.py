import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class TestAlmostEqual(_GenericTest):

    def setup_method(self):
        self._assert_func = assert_almost_equal

    def test_closeness(self):
        self._assert_func(1.499999, 0.0, decimal=0)
        assert_raises(AssertionError, lambda: self._assert_func(1.5, 0.0, decimal=0))
        self._assert_func([1.499999], [0.0], decimal=0)
        assert_raises(AssertionError, lambda: self._assert_func([1.5], [0.0], decimal=0))

    def test_nan_item(self):
        self._assert_func(np.nan, np.nan)
        assert_raises(AssertionError, lambda: self._assert_func(np.nan, 1))
        assert_raises(AssertionError, lambda: self._assert_func(np.nan, np.inf))
        assert_raises(AssertionError, lambda: self._assert_func(np.inf, np.nan))

    def test_inf_item(self):
        self._assert_func(np.inf, np.inf)
        self._assert_func(-np.inf, -np.inf)
        assert_raises(AssertionError, lambda: self._assert_func(np.inf, 1))
        assert_raises(AssertionError, lambda: self._assert_func(-np.inf, np.inf))

    def test_simple_item(self):
        self._test_not_equal(1, 2)

    def test_complex_item(self):
        self._assert_func(complex(1, 2), complex(1, 2))
        self._assert_func(complex(1, np.nan), complex(1, np.nan))
        self._assert_func(complex(np.inf, np.nan), complex(np.inf, np.nan))
        self._test_not_equal(complex(1, np.nan), complex(1, 2))
        self._test_not_equal(complex(np.nan, 1), complex(1, np.nan))
        self._test_not_equal(complex(np.nan, np.inf), complex(np.nan, 2))

    def test_complex(self):
        x = np.array([complex(1, 2), complex(1, np.nan)])
        z = np.array([complex(1, 2), complex(np.nan, 1)])
        y = np.array([complex(1, 2), complex(1, 2)])
        self._assert_func(x, x)
        self._test_not_equal(x, y)
        self._test_not_equal(x, z)

    def test_error_message(self):
        """Check the message is formatted correctly for the decimal value.
           Also check the message when input includes inf or nan (gh12200)"""
        x = np.array([1.00000000001, 2.00000000002, 3.00003])
        y = np.array([1.00000000002, 2.00000000003, 3.00004])
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y, decimal=12)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 3 / 3 (100%)')
        assert_equal(msgs[4], 'Max absolute difference: 1.e-05')
        assert_equal(msgs[5], 'Max relative difference: 3.33328889e-06')
        assert_equal(msgs[6], ' x: array([1.00000000001, 2.00000000002, 3.00003      ])')
        assert_equal(msgs[7], ' y: array([1.00000000002, 2.00000000003, 3.00004      ])')
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 1 / 3 (33.3%)')
        assert_equal(msgs[4], 'Max absolute difference: 1.e-05')
        assert_equal(msgs[5], 'Max relative difference: 3.33328889e-06')
        assert_equal(msgs[6], ' x: array([1.     , 2.     , 3.00003])')
        assert_equal(msgs[7], ' y: array([1.     , 2.     , 3.00004])')
        x = np.array([np.inf, 0])
        y = np.array([np.inf, 1])
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 1 / 2 (50%)')
        assert_equal(msgs[4], 'Max absolute difference: 1.')
        assert_equal(msgs[5], 'Max relative difference: 1.')
        assert_equal(msgs[6], ' x: array([inf,  0.])')
        assert_equal(msgs[7], ' y: array([inf,  1.])')
        x = np.array([1, 2])
        y = np.array([0, 0])
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 2 / 2 (100%)')
        assert_equal(msgs[4], 'Max absolute difference: 2')
        assert_equal(msgs[5], 'Max relative difference: inf')

    def test_error_message_2(self):
        """Check the message is formatted correctly when either x or y is a scalar."""
        x = 2
        y = np.ones(20)
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 20 / 20 (100%)')
        assert_equal(msgs[4], 'Max absolute difference: 1.')
        assert_equal(msgs[5], 'Max relative difference: 1.')
        y = 2
        x = np.ones(20)
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 20 / 20 (100%)')
        assert_equal(msgs[4], 'Max absolute difference: 1.')
        assert_equal(msgs[5], 'Max relative difference: 0.5')

    def test_subclass_that_cannot_be_bool(self):

        class MyArray(np.ndarray):

            def __eq__(self, other):
                return super().__eq__(other).view(np.ndarray)

            def __lt__(self, other):
                return super().__lt__(other).view(np.ndarray)

            def all(self, *args, **kwargs):
                raise NotImplementedError
        a = np.array([1.0, 2.0]).view(MyArray)
        self._assert_func(a, a)