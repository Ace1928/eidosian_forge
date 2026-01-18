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
class TestFromArrow:

    def test_dict(self):
        indices = pyarrow.array([0, 1, 0, 1, 2, 0, None, 2])
        dictionary = pyarrow.array(['first', 'second', 'third'])
        dict_array = pyarrow.DictionaryArray.from_arrays(indices, dictionary)
        at = pyarrow.table({'col1': dict_array, 'col2': [1, 2, 3, 4, 5, 6, 7, 8], 'col3': dict_array})
        pdf = at.to_pandas()
        nchunks = 3
        chunks = split_df_into_chunks(pdf, nchunks)
        at = pyarrow.concat_tables([pyarrow.Table.from_pandas(c) for c in chunks])
        mdf = from_arrow(at)
        at = mdf._query_compiler._modin_frame._partitions[0][0].get()
        assert len(at.column(0).chunks) == nchunks
        mdt = mdf.dtypes.iloc[0]
        pdt = pdf.dtypes.iloc[0]
        assert mdt == 'category'
        assert isinstance(mdt, pandas.CategoricalDtype)
        assert str(mdt) == str(pdt)
        assert type(mdt) is not pandas.CategoricalDtype
        assert mdt._parent is not None
        assert mdt._update_proxy(at, at.column(0)._name) is mdt
        assert mdt._update_proxy(at, at.column(2)._name) is not mdt
        assert type(mdt._update_proxy(at, at.column(2)._name)) != pandas.CategoricalDtype
        assert mdt == pdt
        assert pdt == mdt
        assert repr(mdt) == repr(pdt)
        df_equals(mdf, pdf)
        assert type(mdt._update_proxy(at, at.column(2)._name)) == pandas.CategoricalDtype