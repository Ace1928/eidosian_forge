from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestMultiIndexScalar:

    def test_multiindex_at_get(self):
        df = DataFrame({'a': [1, 2]}, index=[[1, 2], [3, 4]])
        assert df.index.nlevels == 2
        assert df.at[(1, 3), 'a'] == 1
        assert df.loc[(1, 3), 'a'] == 1
        series = df['a']
        assert series.index.nlevels == 2
        assert series.at[1, 3] == 1
        assert series.loc[1, 3] == 1

    @pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
    def test_multiindex_at_set(self):
        df = DataFrame({'a': [1, 2]}, index=[[1, 2], [3, 4]])
        assert df.index.nlevels == 2
        df.at[(1, 3), 'a'] = 3
        assert df.at[(1, 3), 'a'] == 3
        df.loc[(1, 3), 'a'] = 4
        assert df.loc[(1, 3), 'a'] == 4
        series = df['a']
        assert series.index.nlevels == 2
        series.at[1, 3] = 5
        assert series.at[1, 3] == 5
        series.loc[1, 3] = 6
        assert series.loc[1, 3] == 6

    def test_multiindex_at_get_one_level(self):
        s2 = Series((0, 1), index=[[False, True]])
        result = s2.at[False]
        assert result == 0