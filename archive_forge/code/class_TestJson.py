import contextlib
import csv
import inspect
import os
import sys
import unittest.mock as mock
from collections import defaultdict
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict
import fastparquet
import numpy as np
import pandas
import pandas._libs.lib as lib
import pyarrow as pa
import pyarrow.dataset
import pytest
import sqlalchemy as sa
from packaging import version
from pandas._testing import ensure_clean
from pandas.errors import ParserWarning
from scipy import sparse
from modin.config import (
from modin.db_conn import ModinDatabaseConnection, UnsupportedDatabaseException
from modin.pandas.io import from_arrow, from_dask, from_ray, to_pandas
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import (
from .utils import test_data as utils_test_data
from .utils import time_parsing_csv_path
from modin.config import NPartitions
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestJson:

    @pytest.mark.parametrize('pathlike', [False, True])
    @pytest.mark.parametrize('lines', [False, True])
    def test_read_json(self, make_json_file, lines, pathlike):
        unique_filename = make_json_file(lines=lines)
        eval_io(fn_name='read_json', path_or_buf=Path(unique_filename) if pathlike else unique_filename, lines=lines)

    @pytest.mark.parametrize('dtype_backend', [lib.no_default, 'numpy_nullable', 'pyarrow'])
    def test_read_json_dtype_backend(self, make_json_file, dtype_backend):

        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)
        eval_io(fn_name='read_json', path_or_buf=make_json_file(lines=True), lines=True, dtype_backend=dtype_backend, comparator=comparator)

    @pytest.mark.parametrize('storage_options_extra', [{'anon': False}, {'anon': True}, {'key': '123', 'secret': '123'}])
    def test_read_json_s3(self, s3_resource, s3_storage_options, storage_options_extra):
        s3_path = 's3://modin-test/modin-bugs/test_data.json'
        expected_exception = None
        if 'anon' in storage_options_extra:
            expected_exception = PermissionError('Forbidden')
        eval_io(fn_name='read_json', path_or_buf=s3_path, lines=True, orient='records', storage_options=s3_storage_options | storage_options_extra, expected_exception=expected_exception)

    def test_read_json_categories(self):
        eval_io(fn_name='read_json', path_or_buf='modin/tests/pandas/data/test_categories.json', dtype={'one': 'int64', 'two': 'category'})

    def test_read_json_different_columns(self):
        with warns_that_defaulting_to_pandas():
            eval_io(fn_name='read_json', path_or_buf='modin/tests/pandas/data/test_different_columns_in_rows.json', lines=True)

    @pytest.mark.parametrize('data', [json_short_string, json_short_bytes, json_long_string, json_long_bytes])
    def test_read_json_string_bytes(self, data):
        with warns_that_defaulting_to_pandas():
            modin_df = pd.read_json(data)
        if hasattr(data, 'seek'):
            data.seek(0)
        df_equals(modin_df, pandas.read_json(data))

    def test_to_json(self, tmp_path):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(tmp_path, modin_obj=modin_df, pandas_obj=pandas_df, fn='to_json', extension='json')

    @pytest.mark.parametrize('read_mode', ['r', 'rb'])
    def test_read_json_file_handle(self, make_json_file, read_mode):
        with open(make_json_file(), mode=read_mode) as buf:
            df_pandas = pandas.read_json(buf)
            buf.seek(0)
            df_modin = pd.read_json(buf)
            df_equals(df_pandas, df_modin)

    def test_read_json_metadata(self, make_json_file):
        df = pd.read_json(make_json_file(ncols=80, lines=True), lines=True, orient='records')
        parts_width_cached = df._query_compiler._modin_frame._column_widths_cache
        num_splits = len(df._query_compiler._modin_frame._partitions[0])
        parts_width_actual = [len(df._query_compiler._modin_frame._partitions[0][i].get().columns) for i in range(num_splits)]
        assert parts_width_cached == parts_width_actual