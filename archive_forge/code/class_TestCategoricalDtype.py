import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
class TestCategoricalDtype(Base):

    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestCategoricalDtype
        """
        return CategoricalDtype()

    def test_hash_vs_equality(self, dtype):
        dtype2 = CategoricalDtype()
        assert dtype == dtype2
        assert dtype2 == dtype
        assert hash(dtype) == hash(dtype2)

    def test_equality(self, dtype):
        assert dtype == 'category'
        assert is_dtype_equal(dtype, 'category')
        assert 'category' == dtype
        assert is_dtype_equal('category', dtype)
        assert dtype == CategoricalDtype()
        assert is_dtype_equal(dtype, CategoricalDtype())
        assert CategoricalDtype() == dtype
        assert is_dtype_equal(CategoricalDtype(), dtype)
        assert dtype != 'foo'
        assert not is_dtype_equal(dtype, 'foo')
        assert 'foo' != dtype
        assert not is_dtype_equal('foo', dtype)

    def test_construction_from_string(self, dtype):
        result = CategoricalDtype.construct_from_string('category')
        assert is_dtype_equal(dtype, result)
        msg = "Cannot construct a 'CategoricalDtype' from 'foo'"
        with pytest.raises(TypeError, match=msg):
            CategoricalDtype.construct_from_string('foo')

    def test_constructor_invalid(self):
        msg = "Parameter 'categories' must be list-like"
        with pytest.raises(TypeError, match=msg):
            CategoricalDtype('category')
    dtype1 = CategoricalDtype(['a', 'b'], ordered=True)
    dtype2 = CategoricalDtype(['x', 'y'], ordered=False)
    c = Categorical([0, 1], dtype=dtype1)

    @pytest.mark.parametrize('values, categories, ordered, dtype, expected', [[None, None, None, None, CategoricalDtype()], [None, ['a', 'b'], True, None, dtype1], [c, None, None, dtype2, dtype2], [c, ['x', 'y'], False, None, dtype2]])
    def test_from_values_or_dtype(self, values, categories, ordered, dtype, expected):
        result = CategoricalDtype._from_values_or_dtype(values, categories, ordered, dtype)
        assert result == expected

    @pytest.mark.parametrize('values, categories, ordered, dtype', [[None, ['a', 'b'], True, dtype2], [None, ['a', 'b'], None, dtype2], [None, None, True, dtype2]])
    def test_from_values_or_dtype_raises(self, values, categories, ordered, dtype):
        msg = 'Cannot specify `categories` or `ordered` together with `dtype`.'
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype._from_values_or_dtype(values, categories, ordered, dtype)

    def test_from_values_or_dtype_invalid_dtype(self):
        msg = "Cannot not construct CategoricalDtype from <class 'object'>"
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype._from_values_or_dtype(None, None, None, object)

    def test_is_dtype(self, dtype):
        assert CategoricalDtype.is_dtype(dtype)
        assert CategoricalDtype.is_dtype('category')
        assert CategoricalDtype.is_dtype(CategoricalDtype())
        assert not CategoricalDtype.is_dtype('foo')
        assert not CategoricalDtype.is_dtype(np.float64)

    def test_basic(self, dtype):
        msg = 'is_categorical_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_categorical_dtype(dtype)
            factor = Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'])
            s = Series(factor, name='A')
            assert is_categorical_dtype(s.dtype)
            assert is_categorical_dtype(s)
            assert not is_categorical_dtype(np.dtype('float64'))

    def test_tuple_categories(self):
        categories = [(1, 'a'), (2, 'b'), (3, 'c')]
        result = CategoricalDtype(categories)
        assert all(result.categories == categories)

    @pytest.mark.parametrize('categories, expected', [([True, False], True), ([True, False, None], True), ([True, False, 'a', "b'"], False), ([0, 1], False)])
    def test_is_boolean(self, categories, expected):
        cat = Categorical(categories)
        assert cat.dtype._is_boolean is expected
        assert is_bool_dtype(cat) is expected
        assert is_bool_dtype(cat.dtype) is expected

    def test_dtype_specific_categorical_dtype(self):
        expected = 'datetime64[ns]'
        dti = DatetimeIndex([], dtype=expected)
        result = str(Categorical(dti).categories.dtype)
        assert result == expected

    def test_not_string(self):
        assert not is_string_dtype(CategoricalDtype())

    def test_repr_range_categories(self):
        rng = pd.Index(range(3))
        dtype = CategoricalDtype(categories=rng, ordered=False)
        result = repr(dtype)
        expected = 'CategoricalDtype(categories=range(0, 3), ordered=False, categories_dtype=int64)'
        assert result == expected

    def test_update_dtype(self):
        result = CategoricalDtype(['a']).update_dtype(Categorical(['b'], ordered=True))
        expected = CategoricalDtype(['b'], ordered=True)
        assert result == expected

    def test_repr(self):
        cat = Categorical(pd.Index([1, 2, 3], dtype='int32'))
        result = cat.dtype.__repr__()
        expected = 'CategoricalDtype(categories=[1, 2, 3], ordered=False, categories_dtype=int32)'
        assert result == expected