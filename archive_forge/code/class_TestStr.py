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
class TestStr:

    def test_str(self):
        data = test_data_values[0]
        mdf = pd.DataFrame(data[next(iter(data.keys()))])
        pdf = pandas.DataFrame(data[next(iter(data.keys()))])
        df_equals(mdf, pdf)
        mds = pd.Series(data[next(iter(data.keys()))])
        pds = pandas.Series(data[next(iter(data.keys()))])
        assert str(mds) == str(pds)

    def test_no_cols(self):

        def run_cols(df, **kwargs):
            return df.loc[1]
        run_and_compare(fn=run_cols, data=None, constructor_kwargs={'index': range(5)}, force_lazy=False)