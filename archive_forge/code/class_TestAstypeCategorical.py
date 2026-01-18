import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestAstypeCategorical:

    def test_astype_from_categorical3(self):
        df = DataFrame({'cats': [1, 2, 3, 4, 5, 6], 'vals': [1, 2, 3, 4, 5, 6]})
        cats = Categorical([1, 2, 3, 4, 5, 6])
        exp_df = DataFrame({'cats': cats, 'vals': [1, 2, 3, 4, 5, 6]})
        df['cats'] = df['cats'].astype('category')
        tm.assert_frame_equal(exp_df, df)

    def test_astype_from_categorical4(self):
        df = DataFrame({'cats': ['a', 'b', 'b', 'a', 'a', 'd'], 'vals': [1, 2, 3, 4, 5, 6]})
        cats = Categorical(['a', 'b', 'b', 'a', 'a', 'd'])
        exp_df = DataFrame({'cats': cats, 'vals': [1, 2, 3, 4, 5, 6]})
        df['cats'] = df['cats'].astype('category')
        tm.assert_frame_equal(exp_df, df)

    def test_categorical_astype_to_int(self, any_int_dtype):
        df = DataFrame(data={'col1': pd.array([2.0, 1.0, 3.0])})
        df.col1 = df.col1.astype('category')
        df.col1 = df.col1.astype(any_int_dtype)
        expected = DataFrame({'col1': pd.array([2, 1, 3], dtype=any_int_dtype)})
        tm.assert_frame_equal(df, expected)

    def test_astype_categorical_to_string_missing(self):
        df = DataFrame(['a', 'b', np.nan])
        expected = df.astype(str)
        cat = df.astype('category')
        result = cat.astype(str)
        tm.assert_frame_equal(result, expected)