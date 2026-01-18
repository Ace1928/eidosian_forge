import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class DotSharedTests:

    @pytest.fixture
    def obj(self):
        raise NotImplementedError

    @pytest.fixture
    def other(self) -> DataFrame:
        """
        other is a DataFrame that is indexed so that obj.dot(other) is valid
        """
        raise NotImplementedError

    @pytest.fixture
    def expected(self, obj, other) -> DataFrame:
        """
        The expected result of obj.dot(other)
        """
        raise NotImplementedError

    @classmethod
    def reduced_dim_assert(cls, result, expected):
        """
        Assertion about results with 1 fewer dimension that self.obj
        """
        raise NotImplementedError

    def test_dot_equiv_values_dot(self, obj, other, expected):
        result = obj.dot(other)
        tm.assert_equal(result, expected)

    def test_dot_2d_ndarray(self, obj, other, expected):
        result = obj.dot(other.values)
        assert np.all(result == expected.values)

    def test_dot_1d_ndarray(self, obj, expected):
        row = obj.iloc[0] if obj.ndim == 2 else obj
        result = obj.dot(row.values)
        expected = obj.dot(row)
        self.reduced_dim_assert(result, expected)

    def test_dot_series(self, obj, other, expected):
        result = obj.dot(other['1'])
        self.reduced_dim_assert(result, expected['1'])

    def test_dot_series_alignment(self, obj, other, expected):
        result = obj.dot(other.iloc[::-1]['1'])
        self.reduced_dim_assert(result, expected['1'])

    def test_dot_aligns(self, obj, other, expected):
        other2 = other.iloc[::-1]
        result = obj.dot(other2)
        tm.assert_equal(result, expected)

    def test_dot_shape_mismatch(self, obj):
        msg = 'Dot product shape mismatch'
        with pytest.raises(Exception, match=msg):
            obj.dot(obj.values[:3])

    def test_dot_misaligned(self, obj, other):
        msg = 'matrices are not aligned'
        with pytest.raises(ValueError, match=msg):
            obj.dot(other.T)