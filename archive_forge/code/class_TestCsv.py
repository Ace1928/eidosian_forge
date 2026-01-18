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
@pytest.mark.usefixtures('TestReadCSVFixture')
@pytest.mark.skipif(IsExperimental.get() and StorageFormat.get() == 'Pyarrow', reason='Segmentation fault; see PR #2347 ffor details')
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestCsv:

    @pytest.mark.parametrize('sep', ['_', ',', '.'])
    @pytest.mark.parametrize('decimal', ['.', '_'])
    @pytest.mark.parametrize('thousands', [None, ',', '_', ' '])
    def test_read_csv_seps(self, make_csv_file, sep, decimal, thousands):
        unique_filename = make_csv_file(delimiter=sep, thousands_separator=thousands, decimal_separator=decimal)
        eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, sep=sep, decimal=decimal, thousands=thousands)

    @pytest.mark.parametrize('sep', [None, '_'])
    @pytest.mark.parametrize('delimiter', ['.', '_'])
    def test_read_csv_seps_except(self, make_csv_file, sep, delimiter):
        unique_filename = make_csv_file(delimiter=delimiter)
        eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, delimiter=delimiter, sep=sep, expected_exception=ValueError('Specified a sep and a delimiter; you can only specify one.'))

    @pytest.mark.parametrize('dtype_backend', [lib.no_default, 'numpy_nullable', 'pyarrow'])
    def test_read_csv_dtype_backend(self, make_csv_file, dtype_backend):
        unique_filename = make_csv_file()

        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)
        eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, dtype_backend=dtype_backend, comparator=comparator)

    @pytest.mark.parametrize('header', ['infer', None, 0])
    @pytest.mark.parametrize('index_col', [None, 'col1'])
    @pytest.mark.parametrize('names', [lib.no_default, ['col1'], ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']])
    @pytest.mark.parametrize('usecols', [None, ['col1'], ['col1', 'col2', 'col6'], [0, 1, 5]])
    @pytest.mark.parametrize('skip_blank_lines', [True, False])
    def test_read_csv_col_handling(self, header, index_col, names, usecols, skip_blank_lines):
        if names is lib.no_default:
            pytest.skip('some parameters combiantions fails: issue #2312')
        if header in ['infer', None] and names is not lib.no_default:
            pytest.skip('Heterogeneous data in a column is not cast to a common type: issue #3346')
        eval_io(fn_name='read_csv', filepath_or_buffer=pytest.csvs_names['test_read_csv_blank_lines'], header=header, index_col=index_col, names=names, usecols=usecols, skip_blank_lines=skip_blank_lines, expected_exception=False)

    @pytest.mark.parametrize('usecols', [lambda col_name: col_name in ['a', 'b', 'e']])
    def test_from_csv_with_callable_usecols(self, usecols):
        fname = 'modin/tests/pandas/data/test_usecols.csv'
        pandas_df = pandas.read_csv(fname, usecols=usecols)
        modin_df = pd.read_csv(fname, usecols=usecols)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize('dtype', [None, True])
    @pytest.mark.parametrize('engine', [None, 'python', 'c'])
    @pytest.mark.parametrize('converters', [None, {'col1': lambda x: np.int64(x) * 10, 'col2': pandas.to_datetime, 'col4': lambda x: x.replace(':', ';')}])
    @pytest.mark.parametrize('skipfooter', [0, 10])
    def test_read_csv_parsing_1(self, dtype, engine, converters, skipfooter):
        if dtype:
            dtype = {col: 'object' for col in pandas.read_csv(pytest.csvs_names['test_read_csv_regular'], nrows=1).columns}
        expected_exception = None
        if engine == 'c' and skipfooter != 0:
            expected_exception = ValueError("the 'c' engine does not support skipfooter")
        eval_io(fn_name='read_csv', expected_exception=expected_exception, check_kwargs_callable=not callable(converters), filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], dtype=dtype, engine=engine, converters=converters, skipfooter=skipfooter)

    @pytest.mark.parametrize('header', ['infer', None, 0])
    @pytest.mark.parametrize('skiprows', [2, lambda x: x % 2, lambda x: x > 25, lambda x: x > 128, np.arange(10, 50), np.arange(10, 50, 2)])
    @pytest.mark.parametrize('nrows', [35, None])
    @pytest.mark.parametrize('names', [[f'c{col_number}' for col_number in range(4)], [f'c{col_number}' for col_number in range(6)], None])
    @pytest.mark.parametrize('encoding', ['latin1', 'windows-1251', None])
    def test_read_csv_parsing_2(self, make_csv_file, request, header, skiprows, nrows, names, encoding):
        if encoding:
            unique_filename = make_csv_file(encoding=encoding)
        else:
            unique_filename = pytest.csvs_names['test_read_csv_regular']
        kwargs = {'filepath_or_buffer': unique_filename, 'header': header, 'skiprows': skiprows, 'nrows': nrows, 'names': names, 'encoding': encoding}
        if Engine.get() != 'Python':
            df = pandas.read_csv(**dict(kwargs, nrows=1))
            if df[df.columns[0]][df.index[0]] in ['c1', 'col1', 'c3', 'col3']:
                pytest.xfail('read_csv incorrect output with float data - issue #2634')
        eval_io(fn_name='read_csv', expected_exception=None, check_kwargs_callable=not callable(skiprows), **kwargs)

    @pytest.mark.parametrize('true_values', [['Yes'], ['Yes', 'true'], None])
    @pytest.mark.parametrize('false_values', [['No'], ['No', 'false'], None])
    @pytest.mark.parametrize('skipfooter', [0, 10])
    @pytest.mark.parametrize('nrows', [35, None])
    def test_read_csv_parsing_3(self, true_values, false_values, skipfooter, nrows):
        xfail_case = (false_values or true_values) and Engine.get() != 'Python' and (StorageFormat.get() != 'Hdk')
        if xfail_case:
            pytest.xfail('modin and pandas dataframes differs - issue #2446')
        expected_exception = None
        if skipfooter != 0 and nrows is not None:
            expected_exception = ValueError("'skipfooter' not supported with 'nrows'")
        eval_io(fn_name='read_csv', expected_exception=expected_exception, filepath_or_buffer=pytest.csvs_names['test_read_csv_yes_no'], true_values=true_values, false_values=false_values, skipfooter=skipfooter, nrows=nrows)

    def test_read_csv_skipinitialspace(self):
        with ensure_clean('.csv') as unique_filename:
            str_initial_spaces = 'col1,col2,col3,col4\n' + 'five,  six,  seven,  eight\n' + '    five,    six,    seven,    eight\n' + 'five, six,  seven,   eight\n'
            eval_io_from_str(str_initial_spaces, unique_filename, skipinitialspace=True)

    @pytest.mark.parametrize('na_values', ['custom_nan', '73'])
    @pytest.mark.parametrize('keep_default_na', [True, False])
    @pytest.mark.parametrize('na_filter', [True, False])
    @pytest.mark.parametrize('verbose', [True, False])
    @pytest.mark.parametrize('skip_blank_lines', [True, False])
    def test_read_csv_nans_handling(self, na_values, keep_default_na, na_filter, verbose, skip_blank_lines):
        eval_io(fn_name='read_csv', filepath_or_buffer=pytest.csvs_names['test_read_csv_nans'], na_values=na_values, keep_default_na=keep_default_na, na_filter=na_filter, verbose=verbose, skip_blank_lines=skip_blank_lines)

    @pytest.mark.parametrize('parse_dates', [True, False, ['col2'], ['col2', 'col4'], [1, 3]])
    @pytest.mark.parametrize('infer_datetime_format', [True, False])
    @pytest.mark.parametrize('keep_date_col', [True, False])
    @pytest.mark.parametrize('date_parser', [lib.no_default, lambda x: pandas.to_datetime(x, format='%Y-%m-%d')], ids=['default', 'format-Ymd'])
    @pytest.mark.parametrize('dayfirst', [True, False])
    @pytest.mark.parametrize('cache_dates', [True, False])
    def test_read_csv_datetime(self, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, request):
        expected_exception = None
        if 'format-Ymd' in request.node.callspec.id and ('parse_dates3' in request.node.callspec.id or 'parse_dates4' in request.node.callspec.id):
            msg = 'time data "00:00:00" doesn\'t match format "%Y-%m-%d", at position 0. You might want to try:\n' + '    - passing `format` if your strings have a consistent format;\n' + "    - passing `format='ISO8601'` if your strings are all ISO8601 " + 'but not necessarily in exactly the same format;\n' + "    - passing `format='mixed'`, and the format will be inferred " + 'for each element individually. You might want to use `dayfirst` ' + 'alongside this.'
            expected_exception = ValueError(msg)
        eval_io(fn_name='read_csv', check_kwargs_callable=not callable(date_parser), expected_exception=expected_exception, filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], parse_dates=parse_dates, infer_datetime_format=infer_datetime_format, keep_date_col=keep_date_col, date_parser=date_parser, dayfirst=dayfirst, cache_dates=cache_dates)

    @pytest.mark.parametrize('date', ['2023-01-01 00:00:01.000000000', '2023'])
    @pytest.mark.parametrize('dtype', [None, 'str', {'id': 'int64'}])
    @pytest.mark.parametrize('parse_dates', [None, [], ['date'], [1]])
    def test_read_csv_dtype_parse_dates(self, date, dtype, parse_dates):
        with ensure_clean('.csv') as filename:
            with open(filename, 'w') as file:
                file.write(f'id,date\n1,{date}')
            eval_io(fn_name='read_csv', filepath_or_buffer=filename, dtype=dtype, parse_dates=parse_dates)

    @pytest.mark.parametrize('iterator', [True, False])
    def test_read_csv_iteration(self, iterator):
        filename = pytest.csvs_names['test_read_csv_regular']
        rdf_reader = pd.read_csv(filename, chunksize=500, iterator=iterator)
        pd_reader = pandas.read_csv(filename, chunksize=500, iterator=iterator)
        for modin_df, pd_df in zip(rdf_reader, pd_reader):
            df_equals(modin_df, pd_df)
        rdf_reader = pd.read_csv(filename, chunksize=1, iterator=iterator)
        pd_reader = pandas.read_csv(filename, chunksize=1, iterator=iterator)
        modin_df = rdf_reader.get_chunk(1)
        pd_df = pd_reader.get_chunk(1)
        df_equals(modin_df, pd_df)
        rdf_reader = pd.read_csv(filename, chunksize=1, iterator=iterator)
        pd_reader = pandas.read_csv(filename, chunksize=1, iterator=iterator)
        modin_df = rdf_reader.read()
        pd_df = pd_reader.read()
        df_equals(modin_df, pd_df)
        if iterator:
            rdf_reader = pd.read_csv(filename, iterator=iterator)
            pd_reader = pandas.read_csv(filename, iterator=iterator)
            modin_df = rdf_reader.read()
            pd_df = pd_reader.read()
            df_equals(modin_df, pd_df)

    @pytest.mark.parametrize('pathlike', [False, True])
    def test_read_csv_encoding_976(self, pathlike):
        file_name = 'modin/tests/pandas/data/issue_976.csv'
        if pathlike:
            file_name = Path(file_name)
        names = [str(i) for i in range(11)]
        kwargs = {'sep': ';', 'names': names, 'encoding': 'windows-1251'}
        df1 = pd.read_csv(file_name, **kwargs)
        df2 = pandas.read_csv(file_name, **kwargs)
        df1 = df1.drop(['4', '5'], axis=1)
        df2 = df2.drop(['4', '5'], axis=1)
        df_equals(df1, df2)

    @pytest.mark.parametrize('compression', ['infer', 'gzip', 'bz2', 'xz', 'zip'])
    @pytest.mark.parametrize('encoding', [None, 'latin8', 'utf16'])
    @pytest.mark.parametrize('engine', [None, 'python', 'c'])
    def test_read_csv_compression(self, make_csv_file, compression, encoding, engine):
        unique_filename = make_csv_file(encoding=encoding, compression=compression)
        expected_exception = None
        if encoding == 'utf16' and compression in ('bz2', 'xz'):
            expected_exception = UnicodeError('UTF-16 stream does not start with BOM')
        eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, compression=compression, encoding=encoding, engine=engine, expected_exception=expected_exception)

    @pytest.mark.parametrize('encoding', [None, 'ISO-8859-1', 'latin1', 'iso-8859-1', 'cp1252', 'utf8', pytest.param('unicode_escape', marks=pytest.mark.skipif(condition=sys.version_info < (3, 9), reason='https://bugs.python.org/issue45461')), 'raw_unicode_escape', 'utf_16_le', 'utf_16_be', 'utf32', 'utf_32_le', 'utf_32_be', 'utf-8-sig'])
    def test_read_csv_encoding(self, make_csv_file, encoding):
        unique_filename = make_csv_file(encoding=encoding)
        eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, encoding=encoding)

    @pytest.mark.parametrize('thousands', [None, ',', '_', ' '])
    @pytest.mark.parametrize('decimal', ['.', '_'])
    @pytest.mark.parametrize('lineterminator', [None, 'x', '\n'])
    @pytest.mark.parametrize('escapechar', [None, 'd', 'x'])
    @pytest.mark.parametrize('dialect', ['test_csv_dialect', 'use_dialect_name', None])
    def test_read_csv_file_format(self, make_csv_file, thousands, decimal, lineterminator, escapechar, dialect):
        if dialect:
            test_csv_dialect_params = {'delimiter': '_', 'doublequote': False, 'escapechar': '\\', 'quotechar': 'd', 'quoting': csv.QUOTE_ALL}
            csv.register_dialect(dialect, **test_csv_dialect_params)
            if dialect != 'use_dialect_name':
                dialect = csv.get_dialect(dialect)
            unique_filename = make_csv_file(**test_csv_dialect_params)
        else:
            unique_filename = make_csv_file(thousands_separator=thousands, decimal_separator=decimal, escapechar=escapechar, lineterminator=lineterminator)
        if StorageFormat.get() == 'Hdk' and escapechar is not None and (lineterminator is None) and (thousands is None) and (decimal == '.'):
            with open(unique_filename, 'r') as f:
                if any((line.find(f',"{escapechar}') != -1 for _, line in enumerate(f))):
                    pytest.xfail('Tests with this character sequence fail due to #5649')
        expected_exception = None
        if dialect is None:
            expected_exception = False
        eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, thousands=thousands, decimal=decimal, lineterminator=lineterminator, escapechar=escapechar, dialect=dialect, expected_exception=expected_exception)

    @pytest.mark.parametrize('quoting', [csv.QUOTE_ALL, csv.QUOTE_MINIMAL, csv.QUOTE_NONNUMERIC, csv.QUOTE_NONE])
    @pytest.mark.parametrize('quotechar', ['"', '_', 'd'])
    @pytest.mark.parametrize('doublequote', [True, False])
    @pytest.mark.parametrize('comment', [None, '#', 'x'])
    def test_read_csv_quoting(self, make_csv_file, quoting, quotechar, doublequote, comment):
        use_escapechar = not doublequote and quotechar != '"' and (quoting != csv.QUOTE_NONE)
        escapechar = '\\' if use_escapechar else None
        unique_filename = make_csv_file(quoting=quoting, quotechar=quotechar, doublequote=doublequote, escapechar=escapechar, comment_col_char=comment)
        eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, quoting=quoting, quotechar=quotechar, doublequote=doublequote, escapechar=escapechar, comment=comment)

    @pytest.mark.skip(reason='https://github.com/modin-project/modin/issues/6239')
    @pytest.mark.parametrize('on_bad_lines', ['error', 'warn', 'skip', None])
    def test_read_csv_error_handling(self, on_bad_lines):
        raise_exception_case = on_bad_lines is not None
        if not raise_exception_case and Engine.get() not in ['Python'] and (StorageFormat.get() != 'Hdk'):
            pytest.xfail("read_csv doesn't raise `bad lines` exceptions - issue #2500")
        eval_io(fn_name='read_csv', filepath_or_buffer=pytest.csvs_names['test_read_csv_bad_lines'], on_bad_lines=on_bad_lines)

    @pytest.mark.parametrize('float_precision', [None, 'high', 'legacy', 'round_trip'])
    def test_python_engine_float_precision_except(self, float_precision):
        expected_exception = None
        if float_precision is not None:
            expected_exception = ValueError("The 'float_precision' option is not supported with the 'python' engine")
        eval_io(fn_name='read_csv', filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], engine='python', float_precision=float_precision, expected_exception=expected_exception)

    @pytest.mark.parametrize('low_memory', [False, True])
    def test_python_engine_low_memory_except(self, low_memory):
        expected_exception = None
        if not low_memory:
            expected_exception = ValueError("The 'low_memory' option is not supported with the 'python' engine")
        eval_io(fn_name='read_csv', filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], engine='python', low_memory=low_memory, expected_exception=expected_exception)

    @pytest.mark.parametrize('delim_whitespace', [True, False])
    def test_delim_whitespace(self, delim_whitespace, tmp_path):
        if StorageFormat.get() == 'Hdk' and delim_whitespace:
            pytest.xfail(reason='https://github.com/modin-project/modin/issues/6999')
        str_delim_whitespaces = 'col1 col2  col3   col4\n5 6   7  8\n9  10    11 12\n'
        unique_filename = get_unique_filename(data_dir=tmp_path)
        eval_io_from_str(str_delim_whitespaces, unique_filename, delim_whitespace=delim_whitespace)

    @pytest.mark.parametrize('engine', ['c'])
    @pytest.mark.parametrize('delimiter', [',', ' '])
    @pytest.mark.parametrize('low_memory', [True, False])
    @pytest.mark.parametrize('memory_map', [True, False])
    @pytest.mark.parametrize('float_precision', [None, 'high', 'round_trip'])
    def test_read_csv_internal(self, make_csv_file, engine, delimiter, low_memory, memory_map, float_precision):
        unique_filename = make_csv_file(delimiter=delimiter)
        eval_io(filepath_or_buffer=unique_filename, fn_name='read_csv', engine=engine, delimiter=delimiter, low_memory=low_memory, memory_map=memory_map, float_precision=float_precision)

    @pytest.mark.parametrize('nrows', [2, None])
    def test_read_csv_bad_quotes(self, nrows):
        csv_bad_quotes = '1, 2, 3, 4\none, two, three, four\nfive, "six", seven, "eight\n'
        with ensure_clean('.csv') as unique_filename:
            eval_io_from_str(csv_bad_quotes, unique_filename, nrows=nrows)

    def test_read_csv_categories(self):
        eval_io(fn_name='read_csv', filepath_or_buffer='modin/tests/pandas/data/test_categories.csv', names=['one', 'two'], dtype={'one': 'int64', 'two': 'category'})

    @pytest.mark.parametrize('encoding', [None, 'utf-8'])
    @pytest.mark.parametrize('encoding_errors', ['strict', 'ignore'])
    @pytest.mark.parametrize('parse_dates', [pytest.param(value, id=id) for id, value in parse_dates_values_by_id.items()])
    @pytest.mark.parametrize('index_col', [None, 0, 5])
    @pytest.mark.parametrize('header', ['infer', 0])
    @pytest.mark.parametrize('names', [None, ['timestamp', 'year', 'month', 'date', 'symbol', 'high', 'low', 'open', 'close', 'spread', 'volume']])
    @pytest.mark.exclude_in_sanity
    def test_read_csv_parse_dates(self, names, header, index_col, parse_dates, encoding, encoding_errors, request):
        if names is not None and header == 'infer':
            pytest.xfail('read_csv with Ray engine works incorrectly with date data and names parameter provided - issue #2509')
        expected_exception = None
        if 'nonexistent_int_column' in request.node.callspec.id:
            expected_exception = IndexError('list index out of range')
        elif 'nonexistent_string_column' in request.node.callspec.id:
            expected_exception = ValueError("Missing column provided to 'parse_dates': 'z'")
        if StorageFormat.get() == 'Hdk' and 'names1-0-None-nonexistent_string_column-strict-None' in request.node.callspec.id:
            expected_exception = False
        eval_io(fn_name='read_csv', expected_exception=expected_exception, filepath_or_buffer=time_parsing_csv_path, names=names, header=header, index_col=index_col, parse_dates=parse_dates, encoding=encoding, encoding_errors=encoding_errors)

    @pytest.mark.parametrize('storage_options', [{'anon': False}, {'anon': True}, {'key': '123', 'secret': '123'}, None])
    @pytest.mark.xfail(reason='S3 file gone missing, see https://github.com/modin-project/modin/issues/4875')
    def test_read_csv_s3(self, storage_options):
        eval_io(fn_name='read_csv', filepath_or_buffer='s3://noaa-ghcn-pds/csv/1788.csv', storage_options=storage_options)

    def test_read_csv_s3_issue4658(self):
        eval_io(fn_name='read_csv', filepath_or_buffer='s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-01.csv', nrows=10, storage_options={'anon': True})

    @pytest.mark.parametrize('names', [list('XYZ'), None])
    @pytest.mark.parametrize('skiprows', [1, 2, 3, 4, None])
    def test_read_csv_skiprows_names(self, names, skiprows):
        eval_io(fn_name='read_csv', filepath_or_buffer='modin/tests/pandas/data/issue_2239.csv', names=names, skiprows=skiprows)

    def _has_pandas_fallback_reason(self):
        return Engine.get() != 'Python' and StorageFormat.get() != 'Hdk'

    def test_read_csv_default_to_pandas(self):
        if self._has_pandas_fallback_reason():
            warning_suffix = 'buffers'
        else:
            warning_suffix = ''
        with warns_that_defaulting_to_pandas(suffix=warning_suffix):
            with open(pytest.csvs_names['test_read_csv_regular'], 'r') as _f:
                pd.read_csv(StringIO(_f.read()))

    def test_read_csv_url(self):
        eval_io(fn_name='read_csv', filepath_or_buffer='https://raw.githubusercontent.com/modin-project/modin/master/modin/tests/pandas/data/blah.csv', usecols=[0, 1, 2, 3] if StorageFormat.get() == 'Hdk' else None)

    @pytest.mark.parametrize('nrows', [21, 5, None])
    @pytest.mark.parametrize('skiprows', [4, 1, 500, None])
    def test_read_csv_newlines_in_quotes(self, nrows, skiprows):
        expected_exception = None
        if skiprows == 500:
            expected_exception = pandas.errors.EmptyDataError('No columns to parse from file')
        eval_io(fn_name='read_csv', expected_exception=expected_exception, filepath_or_buffer='modin/tests/pandas/data/newlines.csv', nrows=nrows, skiprows=skiprows, cast_to_str=StorageFormat.get() != 'Hdk')

    @pytest.mark.parametrize('skiprows', [None, 0, [], [1, 2], np.arange(0, 2)])
    def test_read_csv_skiprows_with_usecols(self, skiprows):
        usecols = {'float_data': 'float64'}
        expected_exception = None
        if isinstance(skiprows, np.ndarray):
            expected_exception = ValueError("Usecols do not match columns, columns expected but not found: ['float_data']")
        eval_io(fn_name='read_csv', expected_exception=expected_exception, filepath_or_buffer='modin/tests/pandas/data/issue_4543.csv', skiprows=skiprows, usecols=usecols.keys(), dtype=usecols)

    def test_read_csv_sep_none(self):
        eval_io(fn_name='read_csv', modin_warning=ParserWarning, filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], sep=None)

    def test_read_csv_incorrect_data(self):
        eval_io(fn_name='read_csv', filepath_or_buffer='modin/tests/pandas/data/test_categories.json')

    @pytest.mark.parametrize('kwargs', [{'names': [5, 1, 3, 4, 2, 6]}, {'names': [0]}, {'names': None, 'usecols': [1, 0, 2]}, {'names': [3, 1, 2, 5], 'usecols': [4, 1, 3, 2]}])
    def test_read_csv_names_neq_num_cols(self, kwargs):
        eval_io(fn_name='read_csv', filepath_or_buffer='modin/tests/pandas/data/issue_2074.csv', **kwargs)

    def test_read_csv_wrong_path(self):
        expected_exception = FileNotFoundError(2, 'No such file or directory')
        if StorageFormat.get() == 'Hdk':
            expected_exception = False
        eval_io(fn_name='read_csv', expected_exception=expected_exception, filepath_or_buffer='/some/wrong/path.csv')

    @pytest.mark.parametrize('extension', [None, 'csv', 'csv.gz'])
    @pytest.mark.parametrize('sep', [' '])
    @pytest.mark.parametrize('header', [False, True, 'sfx-'])
    @pytest.mark.parametrize('mode', ['w', 'wb+'])
    @pytest.mark.parametrize('idx_name', [None, 'Index'])
    @pytest.mark.parametrize('index', [True, False, 'New index'])
    @pytest.mark.parametrize('index_label', [None, False, 'New index'])
    @pytest.mark.parametrize('columns', [None, ['col1', 'col3', 'col5']])
    @pytest.mark.exclude_in_sanity
    @pytest.mark.skipif(condition=Engine.get() == 'Unidist' and os.name == 'nt', reason='https://github.com/modin-project/modin/issues/6846')
    def test_to_csv(self, tmp_path, extension, sep, header, mode, idx_name, index, index_label, columns):
        pandas_df = generate_dataframe(idx_name=idx_name)
        modin_df = pd.DataFrame(pandas_df)
        if isinstance(header, str):
            if columns is None:
                header = [f'{header}{c}' for c in modin_df.columns]
            else:
                header = [f'{header}{c}' for c in columns]
        eval_to_csv_file(tmp_path, modin_obj=modin_df, pandas_obj=pandas_df, extension=extension, sep=sep, header=header, mode=mode, index=index, index_label=index_label, columns=columns)

    @pytest.mark.skipif(condition=Engine.get() == 'Unidist' and os.name == 'nt', reason='https://github.com/modin-project/modin/issues/6846')
    def test_dataframe_to_csv(self, tmp_path):
        pandas_df = pandas.read_csv(pytest.csvs_names['test_read_csv_regular'])
        modin_df = pd.DataFrame(pandas_df)
        eval_to_csv_file(tmp_path, modin_obj=modin_df, pandas_obj=pandas_df, extension='csv')

    @pytest.mark.skipif(condition=Engine.get() == 'Unidist' and os.name == 'nt', reason='https://github.com/modin-project/modin/issues/6846')
    def test_series_to_csv(self, tmp_path):
        pandas_s = pandas.read_csv(pytest.csvs_names['test_read_csv_regular'], usecols=['col1']).squeeze()
        modin_s = pd.Series(pandas_s)
        eval_to_csv_file(tmp_path, modin_obj=modin_s, pandas_obj=pandas_s, extension='csv')

    def test_read_csv_within_decorator(self):

        @dummy_decorator()
        def wrapped_read_csv(file, method):
            if method == 'pandas':
                return pandas.read_csv(file)
            if method == 'modin':
                return pd.read_csv(file)
        pandas_df = wrapped_read_csv(pytest.csvs_names['test_read_csv_regular'], method='pandas')
        modin_df = wrapped_read_csv(pytest.csvs_names['test_read_csv_regular'], method='modin')
        if StorageFormat.get() == 'Hdk':
            modin_df, pandas_df = align_datetime_dtypes(modin_df, pandas_df)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize('read_mode', ['r', 'rb'])
    @pytest.mark.parametrize('buffer_start_pos', [0, 10])
    @pytest.mark.parametrize('set_async_read_mode', [False, True], indirect=True)
    def test_read_csv_file_handle(self, read_mode, make_csv_file, buffer_start_pos, set_async_read_mode):
        unique_filename = make_csv_file()
        with open(unique_filename, mode=read_mode) as buffer:
            buffer.seek(buffer_start_pos)
            pandas_df = pandas.read_csv(buffer)
            buffer.seek(buffer_start_pos)
            modin_df = pd.read_csv(buffer)
        df_equals(modin_df, pandas_df)

    def test_unnamed_index(self):

        def get_internal_df(df):
            partition = read_df._query_compiler._modin_frame._partitions[0][0]
            return partition.to_pandas()
        path = 'modin/tests/pandas/data/issue_3119.csv'
        read_df = pd.read_csv(path, index_col=0)
        assert get_internal_df(read_df).index.name is None
        read_df = pd.read_csv(path, index_col=[0, 1])
        for name1, name2 in zip(get_internal_df(read_df).index.names, [None, 'a']):
            assert name1 == name2

    def test_read_csv_empty_frame(self):
        eval_io(fn_name='read_csv', filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], usecols=['col1'], index_col='col1')

    @pytest.mark.parametrize('skiprows', [[x for x in range(10)], [x + 5 for x in range(15)], [x for x in range(10) if x % 2 == 0], [x + 5 for x in range(15) if x % 2 == 0], lambda x: x % 2, lambda x: x > 20, lambda x: x < 20, lambda x: True, lambda x: x in [10, 20], lambda x: x << 10])
    @pytest.mark.parametrize('header', ['infer', None, 0, 1, 150])
    def test_read_csv_skiprows_corner_cases(self, skiprows, header):
        eval_io(fn_name='read_csv', check_kwargs_callable=not callable(skiprows), filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], skiprows=skiprows, header=header, dtype='str', expected_exception=False)

    def test_to_csv_with_index(self, tmp_path):
        cols = 100
        arows = 20000
        keyrange = 100
        values = np.vstack([np.random.choice(keyrange, size=arows), np.random.normal(size=(cols, arows))]).transpose()
        modin_df = pd.DataFrame(values, columns=['key'] + ['avalue' + str(i) for i in range(1, 1 + cols)]).set_index('key')
        pandas_df = pandas.DataFrame(values, columns=['key'] + ['avalue' + str(i) for i in range(1, 1 + cols)]).set_index('key')
        eval_to_csv_file(tmp_path, modin_df, pandas_df, 'csv')

    @pytest.mark.parametrize('set_async_read_mode', [False, True], indirect=True)
    def test_read_csv_issue_5150(self, set_async_read_mode):
        with ensure_clean('.csv') as unique_filename:
            pandas_df = pandas.DataFrame(np.random.randint(0, 100, size=(2 ** 6, 2 ** 6)))
            pandas_df.to_csv(unique_filename, index=False)
            expected_pandas_df = pandas.read_csv(unique_filename, index_col=False)
            modin_df = pd.read_csv(unique_filename, index_col=False)
            actual_pandas_df = modin_df._to_pandas()
            if AsyncReadMode.get():
                df_equals(expected_pandas_df, actual_pandas_df)
        if not AsyncReadMode.get():
            df_equals(expected_pandas_df, actual_pandas_df)

    @pytest.mark.parametrize('usecols', [None, [0, 1, 2, 3, 4]])
    def test_read_csv_1930(self, usecols):
        eval_io(fn_name='read_csv', filepath_or_buffer='modin/tests/pandas/data/issue_1930.csv', names=['c1', 'c2', 'c3', 'c4', 'c5'], usecols=usecols)