import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
from pandas._config import using_copy_on_write
from pandas._config.config import _get_option
from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io.parquet import (
class TestParquetFastParquet(Base):

    def test_basic(self, fp, df_full):
        df = df_full
        dti = pd.date_range('20130101', periods=3, tz='US/Eastern')
        dti = dti._with_freq(None)
        df['datetime_tz'] = dti
        df['timedelta'] = pd.timedelta_range('1 day', periods=3)
        check_round_trip(df, fp)

    def test_columns_dtypes_invalid(self, fp):
        df = pd.DataFrame({'string': list('abc'), 'int': list(range(1, 4))})
        err = TypeError
        msg = 'Column name must be a string'
        df.columns = [0, 1]
        self.check_error_on_write(df, fp, err, msg)
        df.columns = [b'foo', b'bar']
        self.check_error_on_write(df, fp, err, msg)
        df.columns = [datetime.datetime(2011, 1, 1, 0, 0), datetime.datetime(2011, 1, 1, 1, 1)]
        self.check_error_on_write(df, fp, err, msg)

    def test_duplicate_columns(self, fp):
        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list('aaa')).copy()
        msg = 'Cannot create parquet dataset with duplicate column names'
        self.check_error_on_write(df, fp, ValueError, msg)

    def test_bool_with_none(self, fp):
        df = pd.DataFrame({'a': [True, None, False]})
        expected = pd.DataFrame({'a': [1.0, np.nan, 0.0]}, dtype='float16')
        check_round_trip(df, fp, expected=expected, check_dtype=False)

    def test_unsupported(self, fp):
        df = pd.DataFrame({'a': pd.period_range('2013', freq='M', periods=3)})
        self.check_error_on_write(df, fp, ValueError, None)
        df = pd.DataFrame({'a': ['a', 1, 2.0]})
        msg = "Can't infer object conversion type"
        self.check_error_on_write(df, fp, ValueError, msg)

    def test_categorical(self, fp):
        df = pd.DataFrame({'a': pd.Categorical(list('abc'))})
        check_round_trip(df, fp)

    def test_filter_row_groups(self, fp):
        d = {'a': list(range(3))}
        df = pd.DataFrame(d)
        with tm.ensure_clean() as path:
            df.to_parquet(path, engine=fp, compression=None, row_group_offsets=1)
            result = read_parquet(path, fp, filters=[('a', '==', 0)])
        assert len(result) == 1

    @pytest.mark.single_cpu
    def test_s3_roundtrip(self, df_compat, s3_public_bucket, fp, s3so):
        check_round_trip(df_compat, fp, path=f's3://{s3_public_bucket.name}/fastparquet.parquet', read_kwargs={'storage_options': s3so}, write_kwargs={'compression': None, 'storage_options': s3so})

    def test_partition_cols_supported(self, tmp_path, fp, df_full):
        partition_cols = ['bool', 'int']
        df = df_full
        df.to_parquet(tmp_path, engine='fastparquet', partition_cols=partition_cols, compression=None)
        assert os.path.exists(tmp_path)
        import fastparquet
        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 2

    def test_partition_cols_string(self, tmp_path, fp, df_full):
        partition_cols = 'bool'
        df = df_full
        df.to_parquet(tmp_path, engine='fastparquet', partition_cols=partition_cols, compression=None)
        assert os.path.exists(tmp_path)
        import fastparquet
        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 1

    def test_partition_on_supported(self, tmp_path, fp, df_full):
        partition_cols = ['bool', 'int']
        df = df_full
        df.to_parquet(tmp_path, engine='fastparquet', compression=None, partition_on=partition_cols)
        assert os.path.exists(tmp_path)
        import fastparquet
        actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
        assert len(actual_partition_cols) == 2

    def test_error_on_using_partition_cols_and_partition_on(self, tmp_path, fp, df_full):
        partition_cols = ['bool', 'int']
        df = df_full
        msg = 'Cannot use both partition_on and partition_cols. Use partition_cols for partitioning data'
        with pytest.raises(ValueError, match=msg):
            df.to_parquet(tmp_path, engine='fastparquet', compression=None, partition_on=partition_cols, partition_cols=partition_cols)

    @pytest.mark.skipif(using_copy_on_write(), reason='fastparquet writes into Index')
    def test_empty_dataframe(self, fp):
        df = pd.DataFrame()
        expected = df.copy()
        check_round_trip(df, fp, expected=expected)

    @pytest.mark.skipif(using_copy_on_write(), reason='fastparquet writes into Index')
    def test_timezone_aware_index(self, fp, timezone_aware_date_list):
        idx = 5 * [timezone_aware_date_list]
        df = pd.DataFrame(index=idx, data={'index_as_col': idx})
        expected = df.copy()
        expected.index.name = 'index'
        check_round_trip(df, fp, expected=expected)

    def test_use_nullable_dtypes_not_supported(self, fp):
        df = pd.DataFrame({'a': [1, 2]})
        with tm.ensure_clean() as path:
            df.to_parquet(path)
            with pytest.raises(ValueError, match='not supported for the fastparquet'):
                with tm.assert_produces_warning(FutureWarning):
                    read_parquet(path, engine='fastparquet', use_nullable_dtypes=True)
            with pytest.raises(ValueError, match='not supported for the fastparquet'):
                read_parquet(path, engine='fastparquet', dtype_backend='pyarrow')

    def test_close_file_handle_on_read_error(self):
        with tm.ensure_clean('test.parquet') as path:
            pathlib.Path(path).write_bytes(b'breakit')
            with pytest.raises(Exception, match=''):
                read_parquet(path, engine='fastparquet')
            pathlib.Path(path).unlink(missing_ok=False)

    def test_bytes_file_name(self, engine):
        df = pd.DataFrame(data={'A': [0, 1], 'B': [1, 0]})
        with tm.ensure_clean('test.parquet') as path:
            with open(path.encode(), 'wb') as f:
                df.to_parquet(f)
            result = read_parquet(path, engine=engine)
        tm.assert_frame_equal(result, df)

    def test_filesystem_notimplemented(self):
        pytest.importorskip('fastparquet')
        df = pd.DataFrame(data={'A': [0, 1], 'B': [1, 0]})
        with tm.ensure_clean() as path:
            with pytest.raises(NotImplementedError, match='filesystem is not implemented'):
                df.to_parquet(path, engine='fastparquet', filesystem='foo')
        with tm.ensure_clean() as path:
            pathlib.Path(path).write_bytes(b'foo')
            with pytest.raises(NotImplementedError, match='filesystem is not implemented'):
                read_parquet(path, engine='fastparquet', filesystem='foo')

    def test_invalid_filesystem(self):
        pytest.importorskip('pyarrow')
        df = pd.DataFrame(data={'A': [0, 1], 'B': [1, 0]})
        with tm.ensure_clean() as path:
            with pytest.raises(ValueError, match='filesystem must be a pyarrow or fsspec FileSystem'):
                df.to_parquet(path, engine='pyarrow', filesystem='foo')
        with tm.ensure_clean() as path:
            pathlib.Path(path).write_bytes(b'foo')
            with pytest.raises(ValueError, match='filesystem must be a pyarrow or fsspec FileSystem'):
                read_parquet(path, engine='pyarrow', filesystem='foo')

    def test_unsupported_pa_filesystem_storage_options(self):
        pa_fs = pytest.importorskip('pyarrow.fs')
        df = pd.DataFrame(data={'A': [0, 1], 'B': [1, 0]})
        with tm.ensure_clean() as path:
            with pytest.raises(NotImplementedError, match='storage_options not supported with a pyarrow FileSystem.'):
                df.to_parquet(path, engine='pyarrow', filesystem=pa_fs.LocalFileSystem(), storage_options={'foo': 'bar'})
        with tm.ensure_clean() as path:
            pathlib.Path(path).write_bytes(b'foo')
            with pytest.raises(NotImplementedError, match='storage_options not supported with a pyarrow FileSystem.'):
                read_parquet(path, engine='pyarrow', filesystem=pa_fs.LocalFileSystem(), storage_options={'foo': 'bar'})

    def test_invalid_dtype_backend(self, engine):
        msg = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
        df = pd.DataFrame({'int': list(range(1, 4))})
        with tm.ensure_clean('tmp.parquet') as path:
            df.to_parquet(path)
            with pytest.raises(ValueError, match=msg):
                read_parquet(path, dtype_backend='numpy')

    @pytest.mark.skipif(using_copy_on_write(), reason='fastparquet writes into Index')
    def test_empty_columns(self, fp):
        df = pd.DataFrame(index=pd.Index(['a', 'b', 'c'], name='custom name'))
        expected = pd.DataFrame(index=pd.Index(['a', 'b', 'c'], name='custom name'))
        check_round_trip(df, fp, expected=expected)