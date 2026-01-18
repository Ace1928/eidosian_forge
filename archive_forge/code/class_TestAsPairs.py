import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
class TestAsPairs:

    def test_single_value(self):
        """Test casting for a single value."""
        expected = np.array([[3, 3]] * 10)
        for x in (3, [3], [[3]]):
            result = _as_pairs(x, 10)
            assert_equal(result, expected)
        obj = object()
        assert_equal(_as_pairs(obj, 10), np.array([[obj, obj]] * 10))

    def test_two_values(self):
        """Test proper casting for two different values."""
        expected = np.array([[3, 4]] * 10)
        for x in ([3, 4], [[3, 4]]):
            result = _as_pairs(x, 10)
            assert_equal(result, expected)
        obj = object()
        assert_equal(_as_pairs(['a', obj], 10), np.array([['a', obj]] * 10))
        assert_equal(_as_pairs([[3], [4]], 2), np.array([[3, 3], [4, 4]]))
        assert_equal(_as_pairs([['a'], [obj]], 2), np.array([['a', 'a'], [obj, obj]]))

    def test_with_none(self):
        expected = ((None, None), (None, None), (None, None))
        assert_equal(_as_pairs(None, 3, as_index=False), expected)
        assert_equal(_as_pairs(None, 3, as_index=True), expected)

    def test_pass_through(self):
        """Test if `x` already matching desired output are passed through."""
        expected = np.arange(12).reshape((6, 2))
        assert_equal(_as_pairs(expected, 6), expected)

    def test_as_index(self):
        """Test results if `as_index=True`."""
        assert_equal(_as_pairs([2.6, 3.3], 10, as_index=True), np.array([[3, 3]] * 10, dtype=np.intp))
        assert_equal(_as_pairs([2.6, 4.49], 10, as_index=True), np.array([[3, 4]] * 10, dtype=np.intp))
        for x in (-3, [-3], [[-3]], [-3, 4], [3, -4], [[-3, 4]], [[4, -3]], [[1, 2]] * 9 + [[1, -2]]):
            with pytest.raises(ValueError, match='negative values'):
                _as_pairs(x, 10, as_index=True)

    def test_exceptions(self):
        """Ensure faulty usage is discovered."""
        with pytest.raises(ValueError, match='more dimensions than allowed'):
            _as_pairs([[[3]]], 10)
        with pytest.raises(ValueError, match='could not be broadcast'):
            _as_pairs([[1, 2], [3, 4]], 3)
        with pytest.raises(ValueError, match='could not be broadcast'):
            _as_pairs(np.ones((2, 3)), 3)