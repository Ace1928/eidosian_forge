import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
class TestMethods(base.BaseMethodsTests):
    _combine_le_expected_dtype = 'boolean'

    def test_factorize(self, data_for_grouping):
        labels, uniques = pd.factorize(data_for_grouping, use_na_sentinel=True)
        expected_labels = np.array([0, 0, -1, -1, 1, 1, 0], dtype=np.intp)
        expected_uniques = data_for_grouping.take([0, 4])
        tm.assert_numpy_array_equal(labels, expected_labels)
        self.assert_extension_array_equal(uniques, expected_uniques)

    def test_searchsorted(self, data_for_sorting, as_series):
        data_for_sorting = pd.array([True, False], dtype='boolean')
        b, a = data_for_sorting
        arr = type(data_for_sorting)._from_sequence([a, b])
        if as_series:
            arr = pd.Series(arr)
        assert arr.searchsorted(a) == 0
        assert arr.searchsorted(a, side='right') == 1
        assert arr.searchsorted(b) == 1
        assert arr.searchsorted(b, side='right') == 2
        result = arr.searchsorted(arr.take([0, 1]))
        expected = np.array([0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        sorter = np.array([1, 0])
        assert data_for_sorting.searchsorted(a, sorter=sorter) == 0

    def test_argmin_argmax(self, data_for_sorting, data_missing_for_sorting):
        assert data_for_sorting.argmax() == 0
        assert data_for_sorting.argmin() == 2
        data = data_for_sorting.take([2, 0, 0, 1, 1, 2])
        assert data.argmax() == 1
        assert data.argmin() == 0
        assert data_missing_for_sorting.argmax() == 0
        assert data_missing_for_sorting.argmin() == 2