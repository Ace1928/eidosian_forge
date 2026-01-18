import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas._libs.arrays import NDArrayBacked
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
class TestCategoricalIndex2:

    def test_view_i8(self):
        ci = CategoricalIndex(list('ab') * 50)
        msg = 'When changing to a larger dtype, its size must be a divisor'
        with pytest.raises(ValueError, match=msg):
            ci.view('i8')
        with pytest.raises(ValueError, match=msg):
            ci._data.view('i8')
        ci = ci[:-4]
        res = ci.view('i8')
        expected = ci._data.codes.view('i8')
        tm.assert_numpy_array_equal(res, expected)
        cat = ci._data
        tm.assert_numpy_array_equal(cat.view('i8'), expected)

    @pytest.mark.parametrize('dtype, engine_type', [(np.int8, libindex.Int8Engine), (np.int16, libindex.Int16Engine), (np.int32, libindex.Int32Engine), (np.int64, libindex.Int64Engine)])
    def test_engine_type(self, dtype, engine_type):
        if dtype != np.int64:
            num_uniques = {np.int8: 1, np.int16: 128, np.int32: 32768}[dtype]
            ci = CategoricalIndex(range(num_uniques))
        else:
            ci = CategoricalIndex(range(32768))
            arr = ci.values._ndarray.astype('int64')
            NDArrayBacked.__init__(ci._data, arr, ci.dtype)
        assert np.issubdtype(ci.codes.dtype, dtype)
        assert isinstance(ci._engine, engine_type)

    @pytest.mark.parametrize('func,op_name', [(lambda idx: idx - idx, '__sub__'), (lambda idx: idx + idx, '__add__'), (lambda idx: idx - ['a', 'b'], '__sub__'), (lambda idx: idx + ['a', 'b'], '__add__'), (lambda idx: ['a', 'b'] - idx, '__rsub__'), (lambda idx: ['a', 'b'] + idx, '__radd__')])
    def test_disallow_addsub_ops(self, func, op_name):
        idx = Index(Categorical(['a', 'b']))
        cat_or_list = "'(Categorical|list)' and '(Categorical|list)'"
        msg = '|'.join([f'cannot perform {op_name} with this index type: CategoricalIndex', 'can only concatenate list', f'unsupported operand type\\(s\\) for [\\+-]: {cat_or_list}'])
        with pytest.raises(TypeError, match=msg):
            func(idx)

    def test_method_delegation(self):
        ci = CategoricalIndex(list('aabbca'), categories=list('cabdef'))
        result = ci.set_categories(list('cab'))
        tm.assert_index_equal(result, CategoricalIndex(list('aabbca'), categories=list('cab')))
        ci = CategoricalIndex(list('aabbca'), categories=list('cab'))
        result = ci.rename_categories(list('efg'))
        tm.assert_index_equal(result, CategoricalIndex(list('ffggef'), categories=list('efg')))
        result = ci.rename_categories(lambda x: x.upper())
        tm.assert_index_equal(result, CategoricalIndex(list('AABBCA'), categories=list('CAB')))
        ci = CategoricalIndex(list('aabbca'), categories=list('cab'))
        result = ci.add_categories(['d'])
        tm.assert_index_equal(result, CategoricalIndex(list('aabbca'), categories=list('cabd')))
        ci = CategoricalIndex(list('aabbca'), categories=list('cab'))
        result = ci.remove_categories(['c'])
        tm.assert_index_equal(result, CategoricalIndex(list('aabb') + [np.nan] + ['a'], categories=list('ab')))
        ci = CategoricalIndex(list('aabbca'), categories=list('cabdef'))
        result = ci.as_unordered()
        tm.assert_index_equal(result, ci)
        ci = CategoricalIndex(list('aabbca'), categories=list('cabdef'))
        result = ci.as_ordered()
        tm.assert_index_equal(result, CategoricalIndex(list('aabbca'), categories=list('cabdef'), ordered=True))
        msg = 'cannot use inplace with CategoricalIndex'
        with pytest.raises(ValueError, match=msg):
            ci.set_categories(list('cab'), inplace=True)

    def test_remove_maintains_order(self):
        ci = CategoricalIndex(list('abcdda'), categories=list('abcd'))
        result = ci.reorder_categories(['d', 'c', 'b', 'a'], ordered=True)
        tm.assert_index_equal(result, CategoricalIndex(list('abcdda'), categories=list('dcba'), ordered=True))
        result = result.remove_categories(['c'])
        tm.assert_index_equal(result, CategoricalIndex(['a', 'b', np.nan, 'd', 'd', 'a'], categories=list('dba'), ordered=True))