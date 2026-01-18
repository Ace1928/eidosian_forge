from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
class TestLagmat:

    @classmethod
    def setup_class(cls):
        data = macrodata.load_pandas()
        cls.macro_df = data.data[['year', 'quarter', 'realgdp', 'cpi']]
        cols = list(cls.macro_df.columns)
        cls.realgdp_loc = cols.index('realgdp')
        cls.cpi_loc = cols.index('cpi')
        np.random.seed(12345)
        cls.random_data = np.random.randn(100)
        index = [str(int(yr)) + '-Q' + str(int(qu)) for yr, qu in zip(cls.macro_df.year, cls.macro_df.quarter)]
        cls.macro_df.index = index
        cls.series = cls.macro_df.cpi

    def test_add_lag_insert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, :3], lagmat, nddata[3:, -1]))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3)
        assert_equal(lag_data, results)

    def test_add_lag_noinsert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3, insert=False)
        assert_equal(lag_data, results)

    def test_add_lag_noinsert_atend(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, -1], 3, trim='Both')
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(data, self.cpi_loc, 3, insert=False)
        assert_equal(lag_data, results)
        lag_data2 = tools.add_lag(data, self.cpi_loc, 3, insert=True)
        assert_equal(lag_data2, results)

    def test_add_lag_ndarray(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, :3], lagmat, nddata[3:, -1]))
        lag_data = tools.add_lag(nddata, 2, 3)
        assert_equal(lag_data, results)

    def test_add_lag_noinsert_ndarray(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(nddata, 2, 3, insert=False)
        assert_equal(lag_data, results)

    def test_add_lag_noinsertatend_ndarray(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, -1], 3, trim='Both')
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(nddata, 3, 3, insert=False)
        assert_equal(lag_data, results)
        lag_data2 = tools.add_lag(nddata, -1, 3, insert=True)
        assert_equal(lag_data2, results)

    def test_sep_return(self):
        data = self.random_data
        n = data.shape[0]
        lagmat, leads = stattools.lagmat(data, 3, trim='none', original='sep')
        expected = np.zeros((n + 3, 4))
        for i in range(4):
            expected[i:i + n, i] = data
        expected_leads = expected[:, :1]
        expected_lags = expected[:, 1:]
        assert_equal(expected_lags, lagmat)
        assert_equal(expected_leads, leads)

    def test_add_lag1d(self):
        data = self.random_data
        lagmat = stattools.lagmat(data, 3, trim='Both')
        results = np.column_stack((data[3:], lagmat))
        lag_data = tools.add_lag(data, lags=3, insert=True)
        assert_equal(results, lag_data)
        data = data[:, None]
        lagmat = stattools.lagmat(data, 3, trim='Both')
        results = np.column_stack((data[3:], lagmat))
        lag_data = tools.add_lag(data, lags=3, insert=True)
        assert_equal(results, lag_data)

    def test_add_lag1d_drop(self):
        data = self.random_data
        lagmat = stattools.lagmat(data, 3, trim='Both')
        lag_data = tools.add_lag(data, lags=3, drop=True, insert=True)
        assert_equal(lagmat, lag_data)
        lag_data = tools.add_lag(data, lags=3, drop=True, insert=False)
        assert_equal(lagmat, lag_data)

    def test_add_lag1d_struct(self):
        data = np.zeros(100, dtype=[('variable', float)])
        nddata = self.random_data
        data['variable'] = nddata
        lagmat = stattools.lagmat(nddata, 3, trim='Both', original='in')
        lag_data = tools.add_lag(data, 0, lags=3, insert=True)
        assert_equal(lagmat, lag_data)
        lag_data = tools.add_lag(data, 0, lags=3, insert=False)
        assert_equal(lagmat, lag_data)
        lag_data = tools.add_lag(data, lags=3, insert=True)
        assert_equal(lagmat, lag_data)

    def test_add_lag_1d_drop_struct(self):
        data = np.zeros(100, dtype=[('variable', float)])
        nddata = self.random_data
        data['variable'] = nddata
        lagmat = stattools.lagmat(nddata, 3, trim='Both')
        lag_data = tools.add_lag(data, lags=3, drop=True)
        assert_equal(lagmat, lag_data)

    def test_add_lag_drop_insert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, :2], lagmat, nddata[3:, -1]))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3, drop=True)
        assert_equal(lag_data, results)

    def test_add_lag_drop_noinsert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim='Both')
        results = np.column_stack((nddata[3:, np.array([0, 1, 3])], lagmat))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3, insert=False, drop=True)
        assert_equal(lag_data, results)

    def test_dataframe_without_pandas(self):
        data = self.macro_df
        both = stattools.lagmat(data, 3, trim='both', original='in')
        both_np = stattools.lagmat(data.values, 3, trim='both', original='in')
        assert_equal(both, both_np)
        lags = stattools.lagmat(data, 3, trim='none', original='ex')
        lags_np = stattools.lagmat(data.values, 3, trim='none', original='ex')
        assert_equal(lags, lags_np)
        lags, lead = stattools.lagmat(data, 3, trim='forward', original='sep')
        lags_np, lead_np = stattools.lagmat(data.values, 3, trim='forward', original='sep')
        assert_equal(lags, lags_np)
        assert_equal(lead, lead_np)

    def test_dataframe_both(self):
        data = self.macro_df
        columns = list(data.columns)
        n = data.shape[0]
        values = np.zeros((n + 3, 16))
        values[:n, :4] = data.values
        for lag in range(1, 4):
            new_cols = [col + '.L.' + str(lag) for col in data]
            columns.extend(new_cols)
            values[lag:n + lag, 4 * lag:4 * (lag + 1)] = data.values
        index = data.index
        values = values[:n]
        expected = pd.DataFrame(values, columns=columns, index=index)
        expected = expected.iloc[3:]
        both = stattools.lagmat(self.macro_df, 3, trim='both', original='in', use_pandas=True)
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(self.macro_df, 3, trim='both', original='ex', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 4:])
        lags, lead = stattools.lagmat(self.macro_df, 3, trim='both', original='sep', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 4:])
        assert_frame_equal(lead, expected.iloc[:, :4])

    def test_too_few_observations(self):
        assert_raises(ValueError, stattools.lagmat, self.macro_df, 300, use_pandas=True)
        assert_raises(ValueError, stattools.lagmat, self.macro_df.values, 300)

    def test_unknown_trim(self):
        assert_raises(ValueError, stattools.lagmat, self.macro_df, 3, trim='unknown', use_pandas=True)
        assert_raises(ValueError, stattools.lagmat, self.macro_df.values, 3, trim='unknown')

    def test_dataframe_forward(self):
        data = self.macro_df
        columns = list(data.columns)
        n = data.shape[0]
        values = np.zeros((n + 3, 16))
        values[:n, :4] = data.values
        for lag in range(1, 4):
            new_cols = [col + '.L.' + str(lag) for col in data]
            columns.extend(new_cols)
            values[lag:n + lag, 4 * lag:4 * (lag + 1)] = data.values
        index = data.index
        values = values[:n]
        expected = pd.DataFrame(values, columns=columns, index=index)
        both = stattools.lagmat(self.macro_df, 3, trim='forward', original='in', use_pandas=True)
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(self.macro_df, 3, trim='forward', original='ex', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 4:])
        lags, lead = stattools.lagmat(self.macro_df, 3, trim='forward', original='sep', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 4:])
        assert_frame_equal(lead, expected.iloc[:, :4])

    def test_pandas_errors(self):
        assert_raises(ValueError, stattools.lagmat, self.macro_df, 3, trim='none', use_pandas=True)
        assert_raises(ValueError, stattools.lagmat, self.macro_df, 3, trim='backward', use_pandas=True)
        assert_raises(ValueError, stattools.lagmat, self.series, 3, trim='none', use_pandas=True)
        assert_raises(ValueError, stattools.lagmat, self.series, 3, trim='backward', use_pandas=True)

    def test_series_forward(self):
        expected = pd.DataFrame(index=self.series.index, columns=['cpi', 'cpi.L.1', 'cpi.L.2', 'cpi.L.3'])
        expected['cpi'] = self.series
        for lag in range(1, 4):
            expected['cpi.L.' + str(int(lag))] = self.series.shift(lag)
        expected = expected.fillna(0.0)
        both = stattools.lagmat(self.series, 3, trim='forward', original='in', use_pandas=True)
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(self.series, 3, trim='forward', original='ex', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 1:])
        lags, lead = stattools.lagmat(self.series, 3, trim='forward', original='sep', use_pandas=True)
        assert_frame_equal(lead, expected.iloc[:, :1])
        assert_frame_equal(lags, expected.iloc[:, 1:])

    def test_series_both(self):
        expected = pd.DataFrame(index=self.series.index, columns=['cpi', 'cpi.L.1', 'cpi.L.2', 'cpi.L.3'])
        expected['cpi'] = self.series
        for lag in range(1, 4):
            expected['cpi.L.' + str(int(lag))] = self.series.shift(lag)
        expected = expected.iloc[3:]
        both = stattools.lagmat(self.series, 3, trim='both', original='in', use_pandas=True)
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(self.series, 3, trim='both', original='ex', use_pandas=True)
        assert_frame_equal(lags, expected.iloc[:, 1:])
        lags, lead = stattools.lagmat(self.series, 3, trim='both', original='sep', use_pandas=True)
        assert_frame_equal(lead, expected.iloc[:, :1])
        assert_frame_equal(lags, expected.iloc[:, 1:])

    def test_range_index_columns(self):
        df = pd.DataFrame(np.arange(200).reshape((-1, 2)))
        df.columns = pd.RangeIndex(2)
        result = stattools.lagmat(df, maxlag=2, use_pandas=True)
        assert result.shape == (100, 4)
        assert list(result.columns) == ['0.L.1', '1.L.1', '0.L.2', '1.L.2']

    def test_duplicate_column_names(self):
        df = pd.DataFrame(np.arange(200).reshape((-1, 2)))
        df.columns = [0, '0']
        with pytest.raises(ValueError, match='Columns names must be'):
            stattools.lagmat(df, maxlag=2, use_pandas=True)