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
@pytest.mark.parametrize('engine', ['pyarrow', 'fastparquet'])
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestParquet:

    @pytest.mark.parametrize('columns', [None, ['col1']])
    @pytest.mark.parametrize('row_group_size', [None, 100, 1000, 10000])
    @pytest.mark.parametrize('path_type', [Path, str])
    def test_read_parquet(self, engine, make_parquet_file, columns, row_group_size, path_type):
        self._test_read_parquet(engine=engine, make_parquet_file=make_parquet_file, columns=columns, filters=None, row_group_size=row_group_size, path_type=path_type)

    def _test_read_parquet(self, engine, make_parquet_file, columns, filters, row_group_size, path_type=str, range_index_start=0, range_index_step=1, range_index_name=None, expected_exception=None):
        if engine == 'pyarrow' and filters == [] and (os.name == 'nt'):
            pytest.xfail('Skipping empty filters error case to avoid race condition - see #6460')
        with ensure_clean('.parquet') as unique_filename:
            unique_filename = path_type(unique_filename)
            make_parquet_file(filename=unique_filename, row_group_size=row_group_size, range_index_start=range_index_start, range_index_step=range_index_step, range_index_name=range_index_name)
            eval_io(fn_name='read_parquet', engine=engine, path=unique_filename, columns=columns, filters=filters, expected_exception=expected_exception)

    @pytest.mark.parametrize('dtype_backend', [lib.no_default, 'numpy_nullable', 'pyarrow'])
    def test_read_parquet_dtype_backend(self, engine, make_parquet_file, dtype_backend):
        with ensure_clean('.parquet') as unique_filename:
            make_parquet_file(filename=unique_filename, row_group_size=100)

            def comparator(df1, df2):
                df_equals(df1, df2)
                df_equals(df1.dtypes, df2.dtypes)
            expected_exception = None
            if engine == 'fastparquet':
                expected_exception = ValueError("The 'dtype_backend' argument is not supported for the fastparquet engine")
            eval_io(fn_name='read_parquet', engine=engine, path=unique_filename, dtype_backend=dtype_backend, comparator=comparator, expected_exception=expected_exception)

    def test_read_parquet_no_extension(self, engine, make_parquet_file):
        with ensure_clean('.parquet') as unique_filename:
            no_ext_fname = unique_filename[:unique_filename.index('.parquet')]
            make_parquet_file(filename=no_ext_fname)
            eval_io(fn_name='read_parquet', engine=engine, path=no_ext_fname)

    @pytest.mark.parametrize('filters', [None, [], [('col1', '==', 5)], [('col1', '<=', 215), ('col2', '>=', 35)]])
    def test_read_parquet_filters(self, engine, make_parquet_file, filters):
        expected_exception = None
        if filters == [] and engine == 'pyarrow':
            expected_exception = ValueError('Malformed filters')
        self._test_read_parquet(engine=engine, make_parquet_file=make_parquet_file, columns=None, filters=filters, row_group_size=100, path_type=str, expected_exception=expected_exception)

    @pytest.mark.parametrize('columns', [None, ['col1']])
    @pytest.mark.parametrize('filters', [None, [('col1', '<=', 1000000)], [('col1', '<=', 75), ('col2', '>=', 35)]])
    @pytest.mark.parametrize('range_index_start', [0, 5000])
    @pytest.mark.parametrize('range_index_step', [1, 10])
    @pytest.mark.parametrize('range_index_name', [None, 'my_index'])
    def test_read_parquet_range_index(self, engine, make_parquet_file, columns, filters, range_index_start, range_index_step, range_index_name):
        self._test_read_parquet(engine=engine, make_parquet_file=make_parquet_file, columns=columns, filters=filters, row_group_size=100, path_type=str, range_index_start=range_index_start, range_index_step=range_index_step, range_index_name=range_index_name)

    def test_read_parquet_list_of_files_5698(self, engine, make_parquet_file):
        if engine == 'fastparquet' and os.name == 'nt':
            pytest.xfail(reason='https://github.com/pandas-dev/pandas/issues/51720')
        with ensure_clean('.parquet') as f1, ensure_clean('.parquet') as f2, ensure_clean('.parquet') as f3:
            for f in [f1, f2, f3]:
                make_parquet_file(filename=f)
            eval_io(fn_name='read_parquet', path=[f1, f2, f3], engine=engine)

    def test_read_parquet_indexing_by_column(self, tmp_path, engine, make_parquet_file):
        nrows = MinPartitionSize.get() + 1
        unique_filename = get_unique_filename(extension='parquet', data_dir=tmp_path)
        make_parquet_file(filename=unique_filename, nrows=nrows)
        parquet_df = pd.read_parquet(unique_filename, engine=engine)
        for col in parquet_df.columns:
            parquet_df[col]

    @pytest.mark.parametrize('columns', [None, ['col1']])
    @pytest.mark.parametrize('filters', [None, [('col1', '<=', 3215), ('col2', '>=', 35)]])
    @pytest.mark.parametrize('row_group_size', [None, 100, 1000, 10000])
    @pytest.mark.parametrize('rows_per_file', [[1000] * 40, [0, 0, 40000], [10000, 10000] + [100] * 200])
    @pytest.mark.exclude_in_sanity
    def test_read_parquet_directory(self, engine, make_parquet_dir, columns, filters, row_group_size, rows_per_file):
        self._test_read_parquet_directory(engine=engine, make_parquet_dir=make_parquet_dir, columns=columns, filters=filters, range_index_start=0, range_index_step=1, range_index_name=None, row_group_size=row_group_size, rows_per_file=rows_per_file)

    def _test_read_parquet_directory(self, engine, make_parquet_dir, columns, filters, range_index_start, range_index_step, range_index_name, row_group_size, rows_per_file):
        num_cols = DATASET_SIZE_DICT.get(TestDatasetSize.get(), DATASET_SIZE_DICT['Small'])
        dfs_by_filename = {}
        start_row = 0
        for i, length in enumerate(rows_per_file):
            end_row = start_row + length
            df = pandas.DataFrame({f'col{x + 1}': np.arange(start_row, end_row) for x in range(num_cols)})
            index = pandas.RangeIndex(start=range_index_start, stop=range_index_start + length * range_index_step, step=range_index_step, name=range_index_name)
            if range_index_start == 0 and range_index_step == 1 and (range_index_name is None):
                assert df.index.equals(index)
            else:
                df.index = index
            dfs_by_filename[f'{i}.parquet'] = df
            start_row = end_row
        path = make_parquet_dir(dfs_by_filename, row_group_size)
        with open(os.path.join(path, '_committed_file'), 'w+') as f:
            f.write('testingtesting')
        eval_io(fn_name='read_parquet', engine=engine, path=path, columns=columns, filters=filters)

    @pytest.mark.parametrize('filters', [None, [('col1', '<=', 1000000)], [('col1', '<=', 75), ('col2', '>=', 35)]])
    @pytest.mark.parametrize('range_index_start', [0, 5000])
    @pytest.mark.parametrize('range_index_step', [1, 10])
    @pytest.mark.parametrize('range_index_name', [None, 'my_index'])
    @pytest.mark.parametrize('row_group_size', [None, 20])
    def test_read_parquet_directory_range_index(self, engine, make_parquet_dir, filters, range_index_start, range_index_step, range_index_name, row_group_size):
        self._test_read_parquet_directory(engine=engine, make_parquet_dir=make_parquet_dir, columns=None, filters=filters, range_index_start=range_index_start, range_index_step=range_index_step, range_index_name=range_index_name, row_group_size=row_group_size, rows_per_file=[250] + [0] * 10 + [25] * 10)

    @pytest.mark.parametrize('filters', [None, [('col1', '<=', 1000000)], [('col1', '<=', 75), ('col2', '>=', 35)]])
    @pytest.mark.parametrize('range_index_start', [0, 5000])
    @pytest.mark.parametrize('range_index_step', [1, 10])
    @pytest.mark.parametrize('range_index_name', [None, 'my_index'])
    def test_read_parquet_directory_range_index_consistent_metadata(self, engine, filters, range_index_start, range_index_step, range_index_name, tmp_path):
        num_cols = DATASET_SIZE_DICT.get(TestDatasetSize.get(), DATASET_SIZE_DICT['Small'])
        df = pandas.DataFrame({f'col{x + 1}': np.arange(0, 500) for x in range(num_cols)})
        index = pandas.RangeIndex(start=range_index_start, stop=range_index_start + len(df) * range_index_step, step=range_index_step, name=range_index_name)
        if range_index_start == 0 and range_index_step == 1 and (range_index_name is None):
            assert df.index.equals(index)
        else:
            df.index = index
        path = get_unique_filename(extension=None, data_dir=tmp_path)
        table = pa.Table.from_pandas(df)
        pyarrow.dataset.write_dataset(table, path, format='parquet', max_rows_per_group=35, max_rows_per_file=100)
        with open(os.path.join(path, '_committed_file'), 'w+') as f:
            f.write('testingtesting')
        eval_io(fn_name='read_parquet', engine=engine, path=path, filters=filters)

    @pytest.mark.parametrize('columns', [None, ['col1']])
    @pytest.mark.parametrize('filters', [None, [], [('col1', '==', 5)], [('col1', '<=', 215), ('col2', '>=', 35)]])
    @pytest.mark.parametrize('range_index_start', [0, 5000])
    @pytest.mark.parametrize('range_index_step', [1, 10])
    def test_read_parquet_partitioned_directory(self, tmp_path, make_parquet_file, columns, filters, range_index_start, range_index_step, engine):
        unique_filename = get_unique_filename(extension=None, data_dir=tmp_path)
        make_parquet_file(filename=unique_filename, partitioned_columns=['col1'], range_index_start=range_index_start, range_index_step=range_index_step, range_index_name='my_index')
        expected_exception = None
        if filters == [] and engine == 'pyarrow':
            expected_exception = ValueError('Malformed filters')
        eval_io(fn_name='read_parquet', engine=engine, path=unique_filename, columns=columns, filters=filters, expected_exception=expected_exception)

    @pytest.mark.parametrize('filters', [None, [], [('B', '==', 'a')], [('B', '==', 'a'), ('A', '>=', 50000), ('idx', '<=', 30000), ('idx_categorical', '==', 'y')]])
    def test_read_parquet_pandas_index(self, engine, filters):
        if version.parse(pa.__version__) >= version.parse('12.0.0') and version.parse(pd.__version__) < version.parse('2.0.0') and (engine == 'pyarrow'):
            pytest.xfail('incompatible versions; see #6072')
        pandas_df = pandas.DataFrame({'idx': np.random.randint(0, 100000, size=2000), 'idx_categorical': pandas.Categorical(['y', 'z'] * 1000), 'idx_periodrange': pandas.period_range(start='2017-01-01', periods=2000), 'A': np.random.randint(0, 100000, size=2000), 'B': ['a', 'b'] * 1000, 'C': ['c'] * 2000})
        if version.parse(pa.__version__) >= version.parse('8.0.0'):
            pandas_df['idx_timedelta'] = pandas.timedelta_range(start='1 day', periods=2000)
        if engine == 'pyarrow':
            pandas_df['idx_datetime'] = pandas.date_range(start='1/1/2018', periods=2000)
        for col in pandas_df.columns:
            if col.startswith('idx'):
                if col == 'idx_categorical' and engine == 'fastparquet' and (version.parse(fastparquet.__version__) < version.parse('2023.1.0')):
                    continue
                with ensure_clean('.parquet') as unique_filename:
                    pandas_df.set_index(col).to_parquet(unique_filename)
                    eval_io('read_parquet', path=unique_filename, engine=engine, filters=filters)
        with ensure_clean('.parquet') as unique_filename:
            pandas_df.set_index(['idx', 'A']).to_parquet(unique_filename)
            eval_io('read_parquet', path=unique_filename, engine=engine, filters=filters)

    @pytest.mark.parametrize('filters', [None, [], [('B', '==', 'a')], [('B', '==', 'a'), ('A', '>=', 5), ('idx', '<=', 30000)]])
    def test_read_parquet_pandas_index_partitioned(self, tmp_path, engine, filters):
        pandas_df = pandas.DataFrame({'idx': np.random.randint(0, 100000, size=2000), 'A': np.random.randint(0, 10, size=2000), 'B': ['a', 'b'] * 1000, 'C': ['c'] * 2000})
        unique_filename = get_unique_filename(extension='parquet', data_dir=tmp_path)
        pandas_df.set_index('idx').to_parquet(unique_filename, partition_cols=['A'])
        expected_exception = None
        if filters == [] and engine == 'pyarrow':
            expected_exception = ValueError('Malformed filters')
        eval_io('read_parquet', path=unique_filename, engine=engine, filters=filters, expected_exception=expected_exception)

    def test_read_parquet_hdfs(self, engine):
        eval_io(fn_name='read_parquet', path='modin/tests/pandas/data/hdfs.parquet', engine=engine)

    @pytest.mark.parametrize('path_type', ['object', 'directory', 'url'])
    def test_read_parquet_s3(self, s3_resource, path_type, engine, s3_storage_options):
        s3_path = 's3://modin-test/modin-bugs/test_data.parquet'
        if path_type == 'object':
            import s3fs
            fs = s3fs.S3FileSystem(endpoint_url=s3_storage_options['client_kwargs']['endpoint_url'])
            with fs.open(s3_path, 'rb') as file_obj:
                eval_io('read_parquet', path=file_obj, engine=engine)
        elif path_type == 'directory':
            s3_path = 's3://modin-test/modin-bugs/test_data_dir.parquet'
            eval_io('read_parquet', path=s3_path, storage_options=s3_storage_options, engine=engine)
        else:
            eval_io('read_parquet', path=s3_path, storage_options=s3_storage_options, engine=engine)

    @pytest.mark.parametrize('filters', [None, [], [('idx', '<=', 30000)], [('idx', '<=', 30000), ('A', '>=', 5)]])
    def test_read_parquet_without_metadata(self, tmp_path, engine, filters):
        """Test that Modin can read parquet files not written by pandas."""
        from pyarrow import csv, parquet
        parquet_fname = get_unique_filename(extension='parquet', data_dir=tmp_path)
        csv_fname = get_unique_filename(extension='parquet', data_dir=tmp_path)
        pandas_df = pandas.DataFrame({'idx': np.random.randint(0, 100000, size=2000), 'A': np.random.randint(0, 10, size=2000), 'B': ['a', 'b'] * 1000, 'C': ['c'] * 2000})
        pandas_df.to_csv(csv_fname, index=False)
        t = csv.read_csv(csv_fname)
        parquet.write_table(t, parquet_fname)
        expected_exception = None
        if filters == [] and engine == 'pyarrow':
            expected_exception = ValueError('Malformed filters')
        eval_io('read_parquet', path=parquet_fname, engine=engine, filters=filters, expected_exception=expected_exception)

    def test_read_empty_parquet_file(self, tmp_path, engine):
        test_df = pandas.DataFrame()
        path = tmp_path / 'data'
        path.mkdir()
        test_df.to_parquet(path / 'part-00000.parquet', engine=engine)
        eval_io(fn_name='read_parquet', path=path, engine=engine)

    @pytest.mark.parametrize('compression_kwargs', [pytest.param({}, id='no_compression_kwargs'), pytest.param({'compression': None}, id='compression=None'), pytest.param({'compression': 'gzip'}, id='compression=gzip'), pytest.param({'compression': 'snappy'}, id='compression=snappy'), pytest.param({'compression': 'brotli'}, id='compression=brotli')])
    @pytest.mark.parametrize('extension', ['parquet', '.gz', '.bz2', '.zip', '.xz'])
    def test_to_parquet(self, tmp_path, engine, compression_kwargs, extension):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        parquet_eval_to_file(tmp_path, modin_obj=modin_df, pandas_obj=pandas_df, fn='to_parquet', extension=extension, engine=engine, **compression_kwargs)

    def test_to_parquet_keep_index(self, tmp_path, engine):
        data = {'c0': [0, 1] * 1000, 'c1': [2, 3] * 1000}
        modin_df, pandas_df = create_test_dfs(data)
        modin_df.index.name = 'foo'
        pandas_df.index.name = 'foo'
        parquet_eval_to_file(tmp_path, modin_obj=modin_df, pandas_obj=pandas_df, fn='to_parquet', extension='parquet', index=True, engine=engine)

    def test_to_parquet_s3(self, s3_resource, engine, s3_storage_options):
        modin_path = 's3://modin-test/modin-dir/modin_df.parquet'
        mdf, pdf = create_test_dfs(utils_test_data['int_data'])
        pdf.to_parquet('s3://modin-test/pandas-dir/pandas_df.parquet', engine=engine, storage_options=s3_storage_options)
        mdf.to_parquet(modin_path, engine=engine, storage_options=s3_storage_options)
        df_equals(pandas.read_parquet('s3://modin-test/pandas-dir/pandas_df.parquet', storage_options=s3_storage_options), pd.read_parquet(modin_path, storage_options=s3_storage_options))
        assert not os.path.isdir(modin_path)

    def test_read_parquet_2462(self, tmp_path, engine):
        test_df = pandas.DataFrame({'col1': [['ad_1', 'ad_2'], ['ad_3']]})
        path = tmp_path / 'data'
        path.mkdir()
        test_df.to_parquet(path / 'part-00000.parquet', engine=engine)
        read_df = pd.read_parquet(path, engine=engine)
        df_equals(test_df, read_df)

    def test_read_parquet_5767(self, tmp_path, engine):
        test_df = pandas.DataFrame({'a': [1, 2, 3, 4], 'b': [1, 1, 2, 2]})
        path = tmp_path / 'data'
        path.mkdir()
        file_name = 'modin_issue#0000.parquet'
        test_df.to_parquet(path / file_name, engine=engine, partition_cols=['b'])
        read_df = pd.read_parquet(path / file_name)
        df_equals(test_df, read_df.astype('int64'))

    @pytest.mark.parametrize('index', [False, True])
    def test_read_parquet_6855(self, tmp_path, engine, index):
        if engine == 'fastparquet':
            pytest.skip("integer columns aren't supported")
        test_df = pandas.DataFrame(np.random.rand(10 ** 2, 10))
        path = tmp_path / 'data'
        path.mkdir()
        file_name = 'issue6855.parquet'
        test_df.to_parquet(path / file_name, index=index, engine=engine)
        read_df = pd.read_parquet(path / file_name, engine=engine)
        if not index:
            read_df.columns = pandas.Index(read_df.columns).astype('int64').to_list()
        df_equals(test_df, read_df)

    def test_read_parquet_s3_with_column_partitioning(self, s3_resource, engine, s3_storage_options):
        s3_path = 's3://modin-test/modin-bugs/issue5159.parquet'
        eval_io(fn_name='read_parquet', path=s3_path, engine=engine, storage_options=s3_storage_options)