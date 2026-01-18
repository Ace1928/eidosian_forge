import contextlib
import json
from pathlib import Path
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.experimental.pandas as pd
from modin.config import AsyncReadMode, Engine
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import try_cast_to_pandas
@pytest.mark.usefixtures('TestReadGlobCSVFixture')
@pytest.mark.skipif(Engine.get() not in ('Ray', 'Unidist', 'Dask'), reason=f'{Engine.get()} does not have experimental glob API')
class TestCsvGlob:

    def test_read_multiple_small_csv(self):
        pandas_df = pandas.concat([pandas.read_csv(fname) for fname in pytest.files])
        modin_df = pd.read_csv_glob(pytest.glob_path)
        pandas_df = pandas_df.reset_index(drop=True)
        modin_df = modin_df.reset_index(drop=True)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize('nrows', [35, 100])
    def test_read_multiple_csv_nrows(self, request, nrows):
        pandas_df = pandas.concat([pandas.read_csv(fname) for fname in pytest.files])
        pandas_df = pandas_df.iloc[:nrows, :]
        modin_df = pd.read_csv_glob(pytest.glob_path, nrows=nrows)
        pandas_df = pandas_df.reset_index(drop=True)
        modin_df = modin_df.reset_index(drop=True)
        df_equals(modin_df, pandas_df)

    def test_read_csv_empty_frame(self):
        kwargs = {'usecols': [0], 'index_col': 0}
        modin_df = pd.read_csv_glob(pytest.files[0], **kwargs)
        pandas_df = pandas.read_csv(pytest.files[0], **kwargs)
        df_equals(modin_df, pandas_df)

    def test_read_csv_without_glob(self):
        with pytest.raises(FileNotFoundError):
            with warns_that_defaulting_to_pandas():
                pd.read_csv_glob('s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-', storage_options={'anon': True})

    def test_read_csv_glob_4373(self, tmp_path):
        columns, filename = (['col0'], str(tmp_path / '1x1.csv'))
        df = pd.DataFrame([[1]], columns=columns)
        with warns_that_defaulting_to_pandas() if Engine.get() == 'Dask' else contextlib.nullcontext():
            df.to_csv(filename)
        kwargs = {'filepath_or_buffer': filename, 'usecols': columns}
        modin_df = pd.read_csv_glob(**kwargs)
        pandas_df = pandas.read_csv(**kwargs)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize('parse_dates', [pytest.param(value, id=id) for id, value in parse_dates_values_by_id.items()])
    def test_read_single_csv_with_parse_dates(self, parse_dates):
        try:
            pandas_df = pandas.read_csv(time_parsing_csv_path, parse_dates=parse_dates)
        except Exception as pandas_exception:
            with pytest.raises(Exception) as modin_exception:
                modin_df = pd.read_csv_glob(time_parsing_csv_path, parse_dates=parse_dates)
                try_cast_to_pandas(modin_df)
            assert isinstance(modin_exception.value, type(pandas_exception)), 'Got Modin Exception type {}, but pandas Exception type {} was expected'.format(type(modin_exception.value), type(pandas_exception))
        else:
            modin_df = pd.read_csv_glob(time_parsing_csv_path, parse_dates=parse_dates)
            df_equals(modin_df, pandas_df)