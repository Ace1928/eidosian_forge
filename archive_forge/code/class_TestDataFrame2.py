from copy import deepcopy
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDataFrame2:

    @pytest.mark.parametrize('value', [1, 'True', [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        msg = 'For argument "inplace" expected type bool, received type'
        with pytest.raises(ValueError, match=msg):
            df.copy().rename_axis(mapper={'a': 'x', 'b': 'y'}, axis=1, inplace=value)
        with pytest.raises(ValueError, match=msg):
            df.copy().drop('a', axis=1, inplace=value)
        with pytest.raises(ValueError, match=msg):
            df.copy().fillna(value=0, inplace=value)
        with pytest.raises(ValueError, match=msg):
            df.copy().replace(to_replace=1, value=7, inplace=value)
        with pytest.raises(ValueError, match=msg):
            df.copy().interpolate(inplace=value)
        with pytest.raises(ValueError, match=msg):
            df.copy()._where(cond=df.a > 2, inplace=value)
        with pytest.raises(ValueError, match=msg):
            df.copy().mask(cond=df.a > 2, inplace=value)

    def test_unexpected_keyword(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), columns=['jim', 'joe'])
        ca = pd.Categorical([0, 0, 2, 2, 3, np.nan])
        ts = df['joe'].copy()
        ts[2] = np.nan
        msg = 'unexpected keyword'
        with pytest.raises(TypeError, match=msg):
            df.drop('joe', axis=1, in_place=True)
        with pytest.raises(TypeError, match=msg):
            df.reindex([1, 0], inplace=True)
        with pytest.raises(TypeError, match=msg):
            ca.fillna(0, inplace=True)
        with pytest.raises(TypeError, match=msg):
            ts.fillna(0, in_place=True)