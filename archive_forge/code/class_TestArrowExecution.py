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
class TestArrowExecution:
    data1 = {'a': [1, 2, 3], 'b': [3, 4, 5], 'c': [6, 7, 8]}
    data2 = {'a': [1, 2, 3], 'd': [3, 4, 5], 'e': [6, 7, 8]}
    data3 = {'a': [4, 5, 6], 'b': [6, 7, 8], 'c': [9, 10, 11]}

    def test_drop_rename_concat(self):

        def drop_rename_concat(df1, df2, lib, **kwargs):
            df1 = df1.rename(columns={'a': 'new_a', 'c': 'new_b'})
            df1 = df1.drop(columns='b')
            df2 = df2.rename(columns={'a': 'new_a', 'd': 'new_b'})
            df2 = df2.drop(columns='e')
            return lib.concat([df1, df2], ignore_index=True)
        run_and_compare(drop_rename_concat, data=self.data1, data2=self.data2, force_lazy=False, force_arrow_execute=True)

    def test_drop_row(self):

        def drop_row(df, **kwargs):
            return df.drop(labels=1)
        run_and_compare(drop_row, data=self.data1, force_lazy=False)

    def test_series_pop(self):

        def pop(df, **kwargs):
            col = df['a']
            col.pop(0)
            return col
        run_and_compare(pop, data=self.data1, force_lazy=False)

    def test_empty_transform(self):

        def apply(df, **kwargs):
            return df + 1
        run_and_compare(apply, data={}, force_arrow_execute=True)

    def test_append(self):

        def apply(df1, df2, **kwargs):
            tmp = df1.append(df2)
            return tmp
        run_and_compare(apply, data=self.data1, data2=self.data3, force_arrow_execute=True)