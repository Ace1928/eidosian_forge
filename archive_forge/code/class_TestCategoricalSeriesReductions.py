from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
class TestCategoricalSeriesReductions:

    @pytest.mark.parametrize('function', ['min', 'max'])
    def test_min_max_unordered_raises(self, function):
        cat = Series(Categorical(['a', 'b', 'c', 'd'], ordered=False))
        msg = f'Categorical is not ordered for operation {function}'
        with pytest.raises(TypeError, match=msg):
            getattr(cat, function)()

    @pytest.mark.parametrize('values, categories', [(list('abc'), list('abc')), (list('abc'), list('cba')), (list('abc') + [np.nan], list('cba')), ([1, 2, 3], [3, 2, 1]), ([1, 2, 3, np.nan], [3, 2, 1])])
    @pytest.mark.parametrize('function', ['min', 'max'])
    def test_min_max_ordered(self, values, categories, function):
        cat = Series(Categorical(values, categories=categories, ordered=True))
        result = getattr(cat, function)(skipna=True)
        expected = categories[0] if function == 'min' else categories[2]
        assert result == expected

    @pytest.mark.parametrize('function', ['min', 'max'])
    @pytest.mark.parametrize('skipna', [True, False])
    def test_min_max_ordered_with_nan_only(self, function, skipna):
        cat = Series(Categorical([np.nan], categories=[1, 2], ordered=True))
        result = getattr(cat, function)(skipna=skipna)
        assert result is np.nan

    @pytest.mark.parametrize('function', ['min', 'max'])
    @pytest.mark.parametrize('skipna', [True, False])
    def test_min_max_skipna(self, function, skipna):
        cat = Series(Categorical(['a', 'b', np.nan, 'a'], categories=['b', 'a'], ordered=True))
        result = getattr(cat, function)(skipna=skipna)
        if skipna is True:
            expected = 'b' if function == 'min' else 'a'
            assert result == expected
        else:
            assert result is np.nan