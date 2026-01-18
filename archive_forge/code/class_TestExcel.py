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
class TestExcel:

    @check_file_leaks
    @pytest.mark.parametrize('pathlike', [False, True])
    def test_read_excel(self, pathlike, make_excel_file):
        unique_filename = make_excel_file()
        eval_io(fn_name='read_excel', io=Path(unique_filename) if pathlike else unique_filename)

    @check_file_leaks
    @pytest.mark.parametrize('skiprows', [2, [1, 3], lambda x: x in [0, 2]])
    def test_read_excel_skiprows(self, skiprows, make_excel_file):
        eval_io(fn_name='read_excel', io=make_excel_file(), skiprows=skiprows, check_kwargs_callable=False)

    @check_file_leaks
    @pytest.mark.parametrize('dtype_backend', [lib.no_default, 'numpy_nullable', 'pyarrow'])
    def test_read_excel_dtype_backend(self, make_excel_file, dtype_backend):

        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)
        eval_io(fn_name='read_excel', io=make_excel_file(), dtype_backend=dtype_backend, comparator=comparator)

    @check_file_leaks
    def test_read_excel_engine(self, make_excel_file):
        eval_io(fn_name='read_excel', modin_warning=UserWarning, io=make_excel_file(), engine='openpyxl')

    @check_file_leaks
    def test_read_excel_index_col(self, make_excel_file):
        eval_io(fn_name='read_excel', modin_warning=UserWarning, io=make_excel_file(), index_col=0)

    @check_file_leaks
    def test_read_excel_all_sheets(self, make_excel_file):
        unique_filename = make_excel_file()
        pandas_df = pandas.read_excel(unique_filename, sheet_name=None)
        modin_df = pd.read_excel(unique_filename, sheet_name=None)
        assert isinstance(pandas_df, dict)
        assert isinstance(modin_df, type(pandas_df))
        assert pandas_df.keys() == modin_df.keys()
        for key in pandas_df.keys():
            df_equals(modin_df.get(key), pandas_df.get(key))

    @pytest.mark.xfail(Engine.get() != 'Python' and StorageFormat.get() != 'Hdk', reason='pandas throws the exception. See pandas issue #39250 for more info')
    @check_file_leaks
    def test_read_excel_sheetname_title(self):
        eval_io(fn_name='read_excel', io='modin/tests/pandas/data/excel_sheetname_title.xlsx', expected_exception=False)

    @check_file_leaks
    def test_excel_empty_line(self):
        path = 'modin/tests/pandas/data/test_emptyline.xlsx'
        modin_df = pd.read_excel(path)
        assert str(modin_df)

    @check_file_leaks
    def test_read_excel_empty_rows(self):
        eval_io(fn_name='read_excel', io='modin/tests/pandas/data/test_empty_rows.xlsx')

    @check_file_leaks
    def test_read_excel_border_rows(self):
        eval_io(fn_name='read_excel', io='modin/tests/pandas/data/test_border_rows.xlsx')

    @check_file_leaks
    def test_read_excel_every_other_nan(self):
        eval_io(fn_name='read_excel', io='modin/tests/pandas/data/every_other_row_nan.xlsx')

    @check_file_leaks
    def test_read_excel_header_none(self):
        eval_io(fn_name='read_excel', io='modin/tests/pandas/data/every_other_row_nan.xlsx', header=None)

    @pytest.mark.parametrize('sheet_name', ['Sheet1', 'AnotherSpecialName', 'SpecialName', 'SecondSpecialName', 0, 1, 2, 3])
    @check_file_leaks
    def test_read_excel_sheet_name(self, sheet_name):
        eval_io(fn_name='read_excel', io='modin/tests/pandas/data/modin_error_book.xlsx', sheet_name=sheet_name, comparator_kwargs={'check_dtypes': False})

    def test_ExcelFile(self, make_excel_file):
        unique_filename = make_excel_file()
        modin_excel_file = pd.ExcelFile(unique_filename)
        pandas_excel_file = pandas.ExcelFile(unique_filename)
        try:
            df_equals(modin_excel_file.parse(), pandas_excel_file.parse())
            assert modin_excel_file.io == unique_filename
        finally:
            modin_excel_file.close()
            pandas_excel_file.close()

    def test_ExcelFile_bytes(self, make_excel_file):
        unique_filename = make_excel_file()
        with open(unique_filename, mode='rb') as f:
            content = f.read()
        modin_excel_file = pd.ExcelFile(content)
        pandas_excel_file = pandas.ExcelFile(content)
        df_equals(modin_excel_file.parse(), pandas_excel_file.parse())

    def test_read_excel_ExcelFile(self, make_excel_file):
        unique_filename = make_excel_file()
        with open(unique_filename, mode='rb') as f:
            content = f.read()
        modin_excel_file = pd.ExcelFile(content)
        pandas_excel_file = pandas.ExcelFile(content)
        df_equals(pd.read_excel(modin_excel_file), pandas.read_excel(pandas_excel_file))

    @pytest.mark.parametrize('use_bytes_io', [False, True])
    def test_read_excel_bytes(self, use_bytes_io, make_excel_file):
        unique_filename = make_excel_file()
        with open(unique_filename, mode='rb') as f:
            io_bytes = f.read()
        if use_bytes_io:
            io_bytes = BytesIO(io_bytes)
        eval_io(fn_name='read_excel', io=io_bytes)

    def test_read_excel_file_handle(self, make_excel_file):
        unique_filename = make_excel_file()
        with open(unique_filename, mode='rb') as f:
            eval_io(fn_name='read_excel', io=f)

    @pytest.mark.xfail(strict=False, reason='Flaky test, defaults to pandas')
    def test_to_excel(self, tmp_path):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        unique_filename_modin = get_unique_filename(extension='xlsx', data_dir=tmp_path)
        unique_filename_pandas = get_unique_filename(extension='xlsx', data_dir=tmp_path)
        modin_writer = pandas.ExcelWriter(unique_filename_modin)
        pandas_writer = pandas.ExcelWriter(unique_filename_pandas)
        modin_df.to_excel(modin_writer)
        pandas_df.to_excel(pandas_writer)
        modin_writer.save()
        pandas_writer.save()
        assert assert_files_eq(unique_filename_modin, unique_filename_pandas)

    @check_file_leaks
    def test_read_excel_empty_frame(self, make_excel_file):
        eval_io(fn_name='read_excel', modin_warning=UserWarning, io=make_excel_file(), usecols=[0], index_col=0)