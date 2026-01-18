import contextlib
import json
import os
import warnings
from io import BytesIO, IOBase, TextIOWrapper
from typing import Any, NamedTuple
import fsspec
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from pandas.util._decorators import doc
from modin.config import MinPartitionSize
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
@doc(_doc_pandas_parser_class, data_type='HDF data')
class PandasHDFParser(PandasParser):

    @staticmethod
    @doc(_doc_parse_func, parameters='fname : str, path object, pandas.HDFStore or file-like object\n    Name of the file, path pandas.HDFStore or file-like object to read.')
    def parse(fname, **kwargs):
        kwargs['key'] = kwargs.pop('_key', None)
        num_splits = kwargs.pop('num_splits', None)
        if num_splits is None:
            return pandas.read_hdf(fname, **kwargs)
        df = pandas.read_hdf(fname, **kwargs)
        return _split_result_for_readers(0, num_splits, df) + [len(df.index), df.dtypes]