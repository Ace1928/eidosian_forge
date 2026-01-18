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
class TestLoc:

    def test_loc(self):
        data = [1, 2, 3, 4, 5, 6]
        idx = ['a', 'b', 'c', 'd', 'e', 'f']
        key = ['b', 'c', 'd', 'e']
        mdf = pd.DataFrame(data, index=idx).loc[key]
        pdf = pandas.DataFrame(data, index=idx).loc[key]
        df_equals(mdf, pdf)

    def test_iloc_bool(self):
        data = [1, 2, 3, 4, 5, 6]
        idx = ['a', 'b', 'c', 'd', 'e', 'f']
        key = [False, True, True, True, True, False]
        mdf = pd.DataFrame(data, index=idx).iloc[key]
        pdf = pandas.DataFrame(data, index=idx).iloc[key]
        df_equals(mdf, pdf)

    def test_iloc_int(self):
        data = range(11, 265)
        key = list(range(0, 11)) + list(range(243, 254))
        mdf = pd.DataFrame(data).iloc[key]
        pdf = pandas.DataFrame(data).iloc[key]
        df_equals(mdf, pdf)
        mdf = pd.DataFrame(data).iloc[range(10, 100)]
        pdf = pandas.DataFrame(data).iloc[range(10, 100)]
        df_equals(mdf, pdf)
        data = test_data_values[0]
        mds = pd.Series(data[next(iter(data.keys()))]).iloc[1:]
        pds = pandas.Series(data[next(iter(data.keys()))]).iloc[1:]
        df_equals(mds, pds)

    def test_iloc_issue_6037(self):

        def iloc(df, **kwargs):
            return df.iloc[:-1].dropna()
        run_and_compare(fn=iloc, data={'A': range(1000000)}, force_lazy=False)