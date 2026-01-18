import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
class TestCategoricalAPIWithFactor:

    def test_describe(self):
        factor = Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'], ordered=True)
        desc = factor.describe()
        assert factor.ordered
        exp_index = CategoricalIndex(['a', 'b', 'c'], name='categories', ordered=factor.ordered)
        expected = DataFrame({'counts': [3, 2, 3], 'freqs': [3 / 8.0, 2 / 8.0, 3 / 8.0]}, index=exp_index)
        tm.assert_frame_equal(desc, expected)
        cat = factor.copy()
        cat = cat.set_categories(['a', 'b', 'c', 'd'])
        desc = cat.describe()
        exp_index = CategoricalIndex(list('abcd'), ordered=factor.ordered, name='categories')
        expected = DataFrame({'counts': [3, 2, 3, 0], 'freqs': [3 / 8.0, 2 / 8.0, 3 / 8.0, 0]}, index=exp_index)
        tm.assert_frame_equal(desc, expected)
        cat = Categorical([1, 2, 3, 1, 2, 3, 3, 2, 1, 1, 1])
        desc = cat.describe()
        exp_index = CategoricalIndex([1, 2, 3], ordered=cat.ordered, name='categories')
        expected = DataFrame({'counts': [5, 3, 3], 'freqs': [5 / 11.0, 3 / 11.0, 3 / 11.0]}, index=exp_index)
        tm.assert_frame_equal(desc, expected)
        cat = Categorical([np.nan, 1, 2, 2])
        desc = cat.describe()
        expected = DataFrame({'counts': [1, 2, 1], 'freqs': [1 / 4.0, 2 / 4.0, 1 / 4.0]}, index=CategoricalIndex([1, 2, np.nan], categories=[1, 2], name='categories'))
        tm.assert_frame_equal(desc, expected)