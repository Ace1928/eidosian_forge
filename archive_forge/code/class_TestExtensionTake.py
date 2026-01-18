from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
class TestExtensionTake:

    def test_bounds_check_large(self):
        arr = np.array([1, 2])
        msg = 'indices are out-of-bounds'
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [2, 3], allow_fill=True)
        msg = 'index 2 is out of bounds for( axis 0 with)? size 2'
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [2, 3], allow_fill=False)

    def test_bounds_check_small(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        indexer = [0, -1, -2]
        msg = "'indices' contains values less than allowed \\(-2 < -1\\)"
        with pytest.raises(ValueError, match=msg):
            algos.take(arr, indexer, allow_fill=True)
        result = algos.take(arr, indexer)
        expected = np.array([1, 3, 2], dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('allow_fill', [True, False])
    def test_take_empty(self, allow_fill):
        arr = np.array([], dtype=np.int64)
        result = algos.take(arr, [], allow_fill=allow_fill)
        tm.assert_numpy_array_equal(arr, result)
        msg = '|'.join(['cannot do a non-empty take from an empty axes.', 'indices are out-of-bounds'])
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [0], allow_fill=allow_fill)

    def test_take_na_empty(self):
        result = algos.take(np.array([]), [-1, -1], allow_fill=True, fill_value=0.0)
        expected = np.array([0.0, 0.0])
        tm.assert_numpy_array_equal(result, expected)

    def test_take_coerces_list(self):
        arr = [1, 2, 3]
        msg = 'take accepting non-standard inputs is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.take(arr, [0, 0])
        expected = np.array([1, 1])
        tm.assert_numpy_array_equal(result, expected)