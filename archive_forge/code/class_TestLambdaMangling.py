import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
class TestLambdaMangling:

    def test_basic(self):
        df = DataFrame({'A': [0, 0, 1, 1], 'B': [1, 2, 3, 4]})
        result = df.groupby('A').agg({'B': [lambda x: 0, lambda x: 1]})
        expected = DataFrame({('B', '<lambda_0>'): [0, 0], ('B', '<lambda_1>'): [1, 1]}, index=Index([0, 1], name='A'))
        tm.assert_frame_equal(result, expected)

    def test_mangle_series_groupby(self):
        gr = Series([1, 2, 3, 4]).groupby([0, 0, 1, 1])
        result = gr.agg([lambda x: 0, lambda x: 1])
        exp_data = {'<lambda_0>': [0, 0], '<lambda_1>': [1, 1]}
        expected = DataFrame(exp_data, index=np.array([0, 1]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(reason='GH-26611. kwargs for multi-agg.')
    def test_with_kwargs(self):
        f1 = lambda x, y, b=1: x.sum() + y + b
        f2 = lambda x, y, b=2: x.sum() + y * b
        result = Series([1, 2]).groupby([0, 0]).agg([f1, f2], 0)
        expected = DataFrame({'<lambda_0>': [4], '<lambda_1>': [6]})
        tm.assert_frame_equal(result, expected)
        result = Series([1, 2]).groupby([0, 0]).agg([f1, f2], 0, b=10)
        expected = DataFrame({'<lambda_0>': [13], '<lambda_1>': [30]})
        tm.assert_frame_equal(result, expected)

    def test_agg_with_one_lambda(self):
        df = DataFrame({'kind': ['cat', 'dog', 'cat', 'dog'], 'height': [9.1, 6.0, 9.5, 34.0], 'weight': [7.9, 7.5, 9.9, 198.0]})
        columns = ['height_sqr_min', 'height_max', 'weight_max']
        expected = DataFrame({'height_sqr_min': [82.81, 36.0], 'height_max': [9.5, 34.0], 'weight_max': [9.9, 198.0]}, index=Index(['cat', 'dog'], name='kind'), columns=columns)
        result1 = df.groupby(by='kind').agg(height_sqr_min=pd.NamedAgg(column='height', aggfunc=lambda x: np.min(x ** 2)), height_max=pd.NamedAgg(column='height', aggfunc='max'), weight_max=pd.NamedAgg(column='weight', aggfunc='max'))
        tm.assert_frame_equal(result1, expected)
        result2 = df.groupby(by='kind').agg(height_sqr_min=('height', lambda x: np.min(x ** 2)), height_max=('height', 'max'), weight_max=('weight', 'max'))
        tm.assert_frame_equal(result2, expected)

    def test_agg_multiple_lambda(self):
        df = DataFrame({'kind': ['cat', 'dog', 'cat', 'dog'], 'height': [9.1, 6.0, 9.5, 34.0], 'weight': [7.9, 7.5, 9.9, 198.0]})
        columns = ['height_sqr_min', 'height_max', 'weight_max', 'height_max_2', 'weight_min']
        expected = DataFrame({'height_sqr_min': [82.81, 36.0], 'height_max': [9.5, 34.0], 'weight_max': [9.9, 198.0], 'height_max_2': [9.5, 34.0], 'weight_min': [7.9, 7.5]}, index=Index(['cat', 'dog'], name='kind'), columns=columns)
        result1 = df.groupby(by='kind').agg(height_sqr_min=('height', lambda x: np.min(x ** 2)), height_max=('height', 'max'), weight_max=('weight', 'max'), height_max_2=('height', lambda x: np.max(x)), weight_min=('weight', lambda x: np.min(x)))
        tm.assert_frame_equal(result1, expected)
        result2 = df.groupby(by='kind').agg(height_sqr_min=pd.NamedAgg(column='height', aggfunc=lambda x: np.min(x ** 2)), height_max=pd.NamedAgg(column='height', aggfunc='max'), weight_max=pd.NamedAgg(column='weight', aggfunc='max'), height_max_2=pd.NamedAgg(column='height', aggfunc=lambda x: np.max(x)), weight_min=pd.NamedAgg(column='weight', aggfunc=lambda x: np.min(x)))
        tm.assert_frame_equal(result2, expected)