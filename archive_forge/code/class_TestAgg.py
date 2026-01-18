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
class TestAgg:
    data = {'a': [1, 2, None, None, 1, None], 'b': [10, 20, None, 20, 10, None], 'c': [None, 200, None, 400, 500, 600], 'd': [11, 22, 33, 22, 33, 22], 'e': [True, True, False, True, False, True]}
    int_data = pandas.DataFrame(data).fillna(0).astype('int').to_dict()

    @pytest.mark.parametrize('agg', ['max', 'min', 'sum', 'mean'])
    @pytest.mark.parametrize('skipna', bool_arg_values)
    def test_simple_agg(self, agg, skipna):

        def apply(df, agg, skipna, **kwargs):
            return getattr(df, agg)(skipna=skipna)
        run_and_compare(apply, data=self.data, agg=agg, skipna=skipna, force_lazy=False)

    def test_count_agg(self):

        def apply(df, **kwargs):
            return df.count()
        run_and_compare(apply, data=self.data, force_lazy=False)

    @pytest.mark.parametrize('data', [data, int_data], ids=['nan_data', 'int_data'])
    @pytest.mark.parametrize('cols', ['a', 'd', ['a', 'd']])
    @pytest.mark.parametrize('dropna', [True, False])
    @pytest.mark.parametrize('sort', [True])
    @pytest.mark.parametrize('ascending', [True, False])
    def test_value_counts(self, data, cols, dropna, sort, ascending):

        def value_counts(df, cols, dropna, sort, ascending, **kwargs):
            return df[cols].value_counts(dropna=dropna, sort=sort, ascending=ascending)
        if dropna and pandas.DataFrame(data, columns=cols if is_list_like(cols) else [cols]).isna().any(axis=None):
            pytest.xfail(reason="'dropna' parameter is forcibly disabled in HDK's GroupBy" + 'due to performance issues, you can track this problem at:' + 'https://github.com/modin-project/modin/issues/2896')
        run_and_compare(value_counts, data=data, cols=cols, dropna=dropna, sort=sort, ascending=ascending, comparator=df_equals_with_non_stable_indices)

    @pytest.mark.parametrize('method', ['sum', 'mean', 'max', 'min', 'count', 'nunique'])
    def test_simple_agg_no_default(self, method):

        def applier(df, **kwargs):
            if isinstance(df, pd.DataFrame):
                with pytest.warns(UserWarning) as warns:
                    res = getattr(df, method)()
                for warn in warns.list:
                    message = warn.message.args[0]
                    if 'is_sparse is deprecated' in message or 'Frame contain columns with unsupported data-types' in message or 'Passing a BlockManager to DataFrame is deprecated' in message:
                        continue
                    assert re.match('.*transpose.*defaulting to pandas', message) is not None, f'Expected DataFrame.transpose defaulting to pandas warning, got: {message}'
            else:
                res = getattr(df, method)()
            return res
        run_and_compare(applier, data=self.data, force_lazy=False)

    @pytest.mark.parametrize('data', [data, int_data])
    @pytest.mark.parametrize('dropna', bool_arg_values)
    def test_nunique(self, data, dropna):

        def applier(df, **kwargs):
            return df.nunique(dropna=dropna)
        run_and_compare(applier, data=data, force_lazy=False)