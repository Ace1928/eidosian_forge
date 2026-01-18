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
class TestConstructor:

    @pytest.mark.parametrize('data', [None, {'A': range(10)}, pandas.Series(range(10)), pandas.DataFrame({'A': range(10)})])
    @pytest.mark.parametrize('index', [None, pandas.RangeIndex(10), pandas.RangeIndex(start=10, stop=0, step=-1)])
    @pytest.mark.parametrize('columns', [None, ['A'], ['A', 'B', 'C']])
    @pytest.mark.parametrize('dtype', [None, float])
    def test_raw_data(self, data, index, columns, dtype):
        if isinstance(data, pandas.Series) and data.name is None and (columns is not None) and (len(columns) > 1):
            data = data.copy()
            data.name = 'D'
        mdf, pdf = create_test_dfs(data, index=index, columns=columns, dtype=dtype)
        df_equals(mdf, pdf)

    @pytest.mark.parametrize('index', [None, pandas.Index([1, 2, 3]), pandas.MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3)])])
    def test_shape_hint_detection(self, index):
        df = pd.DataFrame({'a': [1, 2, 3]}, index=index)
        assert df._query_compiler._shape_hint == 'column'
        transposed_data = df._to_pandas().T.to_dict()
        df = pd.DataFrame(transposed_data)
        assert df._query_compiler._shape_hint == 'row'
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]}, index=index)
        assert df._query_compiler._shape_hint is None
        df = pd.DataFrame({'a': [1]}, index=None if index is None else index[:1])
        assert df._query_compiler._shape_hint == 'column'

    def test_shape_hint_detection_from_arrow(self):
        at = pyarrow.Table.from_pydict({'a': [1, 2, 3]})
        df = pd.utils.from_arrow(at)
        assert df._query_compiler._shape_hint == 'column'
        at = pyarrow.Table.from_pydict({'a': [1], 'b': [2], 'c': [3]})
        df = pd.utils.from_arrow(at)
        assert df._query_compiler._shape_hint == 'row'
        at = pyarrow.Table.from_pydict({'a': [1, 2, 3], 'b': [1, 2, 3]})
        df = pd.utils.from_arrow(at)
        assert df._query_compiler._shape_hint is None
        at = pyarrow.Table.from_pydict({'a': [1]})
        df = pd.utils.from_arrow(at)
        assert df._query_compiler._shape_hint == 'column'

    def test_constructor_from_modin_series(self):

        def construct_has_common_projection(lib, df, **kwargs):
            return lib.DataFrame({'col1': df.iloc[:, 0], 'col2': df.iloc[:, 1]})

        def construct_no_common_projection(lib, df1, df2, **kwargs):
            return lib.DataFrame({'col1': df1.iloc[:, 0], 'col2': df2.iloc[:, 0], 'col3': df1.iloc[:, 1]})

        def construct_mixed_data(lib, df1, df2, **kwargs):
            return lib.DataFrame({'col1': df1.iloc[:, 0], 'col2': df2.iloc[:, 0], 'col3': df1.iloc[:, 1], 'col4': np.arange(len(df1))})
        run_and_compare(construct_has_common_projection, data={'a': [1, 2, 3, 4], 'b': [3, 4, 5, 6]})
        run_and_compare(construct_no_common_projection, data={'a': [1, 2, 3, 4], 'b': [3, 4, 5, 6]}, data2={'a': [10, 20, 30, 40]}, force_lazy=False)
        run_and_compare(construct_mixed_data, data={'a': [1, 2, 3, 4], 'b': [3, 4, 5, 6]}, data2={'a': [10, 20, 30, 40]}, force_lazy=False)