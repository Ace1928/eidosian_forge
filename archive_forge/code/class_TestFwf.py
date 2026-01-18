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
class TestFwf:

    @pytest.mark.parametrize('pathlike', [False, True])
    def test_fwf_file(self, make_fwf_file, pathlike):
        fwf_data = 'id8141  360.242940  149.910199 11950.7\n' + 'id1594  444.953632  166.985655 11788.4\n' + 'id1849  364.136849  183.628767 11806.2\n' + 'id1230  413.836124  184.375703 11916.8\n' + 'id1948  502.953953  173.237159 12468.3\n'
        unique_filename = make_fwf_file(fwf_data=fwf_data)
        colspecs = [(0, 6), (8, 20), (21, 33), (34, 43)]
        df = pd.read_fwf(Path(unique_filename) if pathlike else unique_filename, colspecs=colspecs, header=None, index_col=0)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.parametrize('kwargs', [{'colspecs': [(0, 11), (11, 15), (19, 24), (27, 32), (35, 40), (43, 48), (51, 56), (59, 64), (67, 72), (75, 80), (83, 88), (91, 96), (99, 104), (107, 112)], 'names': ['stationID', 'year', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'na_values': ['-9999'], 'index_col': ['stationID', 'year']}, {'widths': [20, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 'names': ['id', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'index_col': [0]}])
    def test_fwf_file_colspecs_widths(self, make_fwf_file, kwargs):
        unique_filename = make_fwf_file()
        modin_df = pd.read_fwf(unique_filename, **kwargs)
        pandas_df = pd.read_fwf(unique_filename, **kwargs)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize('usecols', [['a'], ['a', 'b', 'd'], [0, 1, 3]])
    def test_fwf_file_usecols(self, make_fwf_file, usecols):
        fwf_data = 'a       b           c          d\n' + 'id8141  360.242940  149.910199 11950.7\n' + 'id1594  444.953632  166.985655 11788.4\n' + 'id1849  364.136849  183.628767 11806.2\n' + 'id1230  413.836124  184.375703 11916.8\n' + 'id1948  502.953953  173.237159 12468.3\n'
        eval_io(fn_name='read_fwf', filepath_or_buffer=make_fwf_file(fwf_data=fwf_data), usecols=usecols)

    @pytest.mark.parametrize('dtype_backend', [lib.no_default, 'numpy_nullable', 'pyarrow'])
    def test_read_fwf_dtype_backend(self, make_fwf_file, dtype_backend):
        unique_filename = make_fwf_file()

        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)
        eval_io(fn_name='read_fwf', filepath_or_buffer=unique_filename, dtype_backend=dtype_backend, comparator=comparator)

    def test_fwf_file_chunksize(self, make_fwf_file):
        unique_filename = make_fwf_file()
        rdf_reader = pd.read_fwf(unique_filename, chunksize=5)
        pd_reader = pandas.read_fwf(unique_filename, chunksize=5)
        for modin_df, pd_df in zip(rdf_reader, pd_reader):
            df_equals(modin_df, pd_df)
        rdf_reader = pd.read_fwf(unique_filename, chunksize=1)
        pd_reader = pandas.read_fwf(unique_filename, chunksize=1)
        modin_df = rdf_reader.get_chunk(1)
        pd_df = pd_reader.get_chunk(1)
        df_equals(modin_df, pd_df)
        rdf_reader = pd.read_fwf(unique_filename, chunksize=1)
        pd_reader = pandas.read_fwf(unique_filename, chunksize=1)
        modin_df = rdf_reader.read()
        pd_df = pd_reader.read()
        df_equals(modin_df, pd_df)

    @pytest.mark.parametrize('nrows', [13, None])
    def test_fwf_file_skiprows(self, make_fwf_file, nrows):
        unique_filename = make_fwf_file()
        eval_io(fn_name='read_fwf', filepath_or_buffer=unique_filename, skiprows=2, nrows=nrows)
        eval_io(fn_name='read_fwf', filepath_or_buffer=unique_filename, usecols=[0, 4, 7], skiprows=[2, 5], nrows=nrows)

    def test_fwf_file_index_col(self, make_fwf_file):
        fwf_data = 'a       b           c          d\n' + 'id8141  360.242940  149.910199 11950.7\n' + 'id1594  444.953632  166.985655 11788.4\n' + 'id1849  364.136849  183.628767 11806.2\n' + 'id1230  413.836124  184.375703 11916.8\n' + 'id1948  502.953953  173.237159 12468.3\n'
        eval_io(fn_name='read_fwf', filepath_or_buffer=make_fwf_file(fwf_data=fwf_data), index_col='c')

    def test_fwf_file_skipfooter(self, make_fwf_file):
        eval_io(fn_name='read_fwf', filepath_or_buffer=make_fwf_file(), skipfooter=2)

    def test_fwf_file_parse_dates(self, make_fwf_file):
        dates = pandas.date_range('2000', freq='h', periods=10)
        fwf_data = 'col1 col2        col3 col4'
        for i in range(10, 20):
            fwf_data = fwf_data + '\n{col1}   {col2}  {col3}   {col4}'.format(col1=str(i), col2=str(dates[i - 10].date()), col3=str(i), col4=str(dates[i - 10].time()))
        unique_filename = make_fwf_file(fwf_data=fwf_data)
        eval_io(fn_name='read_fwf', filepath_or_buffer=unique_filename, parse_dates=[['col2', 'col4']])
        eval_io(fn_name='read_fwf', filepath_or_buffer=unique_filename, parse_dates={'time': ['col2', 'col4']})

    @pytest.mark.parametrize('read_mode', ['r', 'rb'])
    def test_read_fwf_file_handle(self, make_fwf_file, read_mode):
        with open(make_fwf_file(), mode=read_mode) as buffer:
            df_pandas = pandas.read_fwf(buffer)
            buffer.seek(0)
            df_modin = pd.read_fwf(buffer)
            df_equals(df_modin, df_pandas)

    def test_read_fwf_empty_frame(self, make_fwf_file):
        kwargs = {'usecols': [0], 'index_col': 0}
        unique_filename = make_fwf_file()
        modin_df = pd.read_fwf(unique_filename, **kwargs)
        pandas_df = pandas.read_fwf(unique_filename, **kwargs)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize('storage_options_extra', [{'anon': False}, {'anon': True}, {'key': '123', 'secret': '123'}])
    def test_read_fwf_s3(self, s3_resource, s3_storage_options, storage_options_extra):
        expected_exception = None
        if 'anon' in storage_options_extra:
            expected_exception = PermissionError('Forbidden')
        eval_io(fn_name='read_fwf', filepath_or_buffer='s3://modin-test/modin-bugs/test_data.fwf', storage_options=s3_storage_options | storage_options_extra, expected_exception=expected_exception)