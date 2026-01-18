import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class TestAssertAllclose:

    def test_simple(self):
        x = 0.001
        y = 1e-09
        assert_allclose(x, y, atol=1)
        assert_raises(AssertionError, assert_allclose, x, y)
        a = np.array([x, y, x, y])
        b = np.array([x, y, x, x])
        assert_allclose(a, b, atol=1)
        assert_raises(AssertionError, assert_allclose, a, b)
        b[-1] = y * (1 + 1e-08)
        assert_allclose(a, b)
        assert_raises(AssertionError, assert_allclose, a, b, rtol=1e-09)
        assert_allclose(6, 10, rtol=0.5)
        assert_raises(AssertionError, assert_allclose, 10, 6, rtol=0.5)

    def test_min_int(self):
        a = np.array([np.iinfo(np.int_).min], dtype=np.int_)
        assert_allclose(a, a)

    def test_report_fail_percentage(self):
        a = np.array([1, 1, 1, 1])
        b = np.array([1, 1, 1, 2])
        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b)
        msg = str(exc_info.value)
        assert_('Mismatched elements: 1 / 4 (25%)\nMax absolute difference: 1\nMax relative difference: 0.5' in msg)

    def test_equal_nan(self):
        a = np.array([np.nan])
        b = np.array([np.nan])
        assert_allclose(a, b, equal_nan=True)

    def test_not_equal_nan(self):
        a = np.array([np.nan])
        b = np.array([np.nan])
        assert_raises(AssertionError, assert_allclose, a, b, equal_nan=False)

    def test_equal_nan_default(self):
        a = np.array([np.nan])
        b = np.array([np.nan])
        assert_array_equal(a, b)
        assert_array_almost_equal(a, b)
        assert_array_less(a, b)
        assert_allclose(a, b)

    def test_report_max_relative_error(self):
        a = np.array([0, 1])
        b = np.array([0, 2])
        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b)
        msg = str(exc_info.value)
        assert_('Max relative difference: 0.5' in msg)

    def test_timedelta(self):
        a = np.array([[1, 2, 3, 'NaT']], dtype='m8[ns]')
        assert_allclose(a, a)

    def test_error_message_unsigned(self):
        """Check the the message is formatted correctly when overflow can occur
           (gh21768)"""
        x = np.asarray([0, 1, 8], dtype='uint8')
        y = np.asarray([4, 4, 4], dtype='uint8')
        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(x, y, atol=3)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[4], 'Max absolute difference: 4')