import os
import re
import numpy as np
import pandas
import pyarrow
import pytest
from pandas._testing import ensure_clean
from pandas.core.dtypes.common import is_list_like
from pyhdk import __version__ as hdk_version
from modin.config import StorageFormat
from modin.tests.interchange.dataframe_protocol.hdk.utils import split_df_into_chunks
from modin.tests.pandas.utils import (
from .utils import ForceHdkImport, eval_io, run_and_compare, set_execution_mode
import modin.pandas as pd
from modin.experimental.core.execution.native.implementations.hdk_on_native.calcite_serializer import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.df_algebra import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.partitioning.partition_manager import (
from modin.pandas.io import from_arrow
from modin.tests.pandas.utils import (
from modin.utils import try_cast_to_pandas
class TestMasks:
    data = {'a': [1, 1, 2, 2, 3], 'b': [None, None, 2, 1, 3], 'c': [3, None, None, 2, 1]}
    cols_values = ['a', ['a', 'b'], ['a', 'b', 'c']]

    @pytest.mark.parametrize('cols', cols_values)
    def test_projection(self, cols):

        def projection(df, cols, **kwargs):
            return df[cols]
        run_and_compare(projection, data=self.data, cols=cols)

    def test_drop(self):

        def drop(df, column_names, **kwargs):
            return df.drop(columns=column_names)
        run_and_compare(drop, data=self.data, column_names='a')
        run_and_compare(drop, data=self.data, column_names=self.data.keys())

    def test_drop_index(self):

        def drop(df, **kwargs):
            return df.drop(df.index[0])
        idx = list(map(str, self.data['a']))
        run_and_compare(drop, data=self.data, constructor_kwargs={'index': idx}, force_lazy=False)

    def test_iloc(self):

        def mask(df, **kwargs):
            return df.iloc[[0, 1]]
        run_and_compare(mask, data=self.data, allow_subqueries=True)

    def test_empty(self):

        def empty(df, **kwargs):
            return df
        run_and_compare(empty, data=None)

    def test_filter(self):

        def filter(df, **kwargs):
            return df[df['a'] == 1]
        run_and_compare(filter, data=self.data)

    def test_filter_with_index(self):

        def filter(df, **kwargs):
            df = df.groupby('a').sum()
            return df[df['b'] > 1]
        run_and_compare(filter, data=self.data)

    def test_filter_proj(self):

        def filter(df, **kwargs):
            df1 = df + 2
            return df1[df['a'] + df1['b'] > 1]
        run_and_compare(filter, data=self.data)

    def test_filter_drop(self):

        def filter(df, **kwargs):
            df = df[['a', 'b']]
            df = df[df['a'] != 1]
            df['a'] = df['a'] * df['b']
            return df
        run_and_compare(filter, data=self.data)

    def test_filter_str_categorical(self):

        def filter(df, **kwargs):
            return df[df['A'] != '']
        data = {'A': ['A', 'B', 'C']}
        run_and_compare(filter, data=data)
        run_and_compare(filter, data=data, constructor_kwargs={'dtype': 'category'})