from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import seasonal_decompose
class TestDecompose:

    @classmethod
    def setup_class(cls):
        data = [-50, 175, 149, 214, 247, 237, 225, 329, 729, 809, 530, 489, 540, 457, 195, 176, 337, 239, 128, 102, 232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184]
        cls.data = pd.DataFrame(data, pd.date_range(start='1/1/1951', periods=len(data), freq=QUARTER_END))

    def test_ndarray(self):
        res_add = seasonal_decompose(self.data.values, period=4)
        assert_almost_equal(res_add.seasonal, SEASONAL, 2)
        assert_almost_equal(res_add.trend, TREND, 2)
        assert_almost_equal(res_add.resid, RANDOM, 3)
        res_mult = seasonal_decompose(np.abs(self.data.values), 'm', period=4)
        assert_almost_equal(res_mult.seasonal, MULT_SEASONAL, 4)
        assert_almost_equal(res_mult.trend, MULT_TREND, 2)
        assert_almost_equal(res_mult.resid, MULT_RANDOM, 4)
        res_add = seasonal_decompose(self.data.values[:-1], period=4)
        seasonal = [68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66]
        trend = [np.nan, np.nan, 159.12, 204.0, 221.25, 245.12, 319.75, 451.5, 561.12, 619.25, 615.62, 548.0, 462.12, 381.12, 316.62, 264.0, 228.38, 210.75, 188.38, 199.0, 207.12, 191.0, 166.88, 72.0, -9.25, -33.12, -36.75, 36.25, 103.0, np.nan, np.nan]
        random = [np.nan, np.nan, 72.538, 64.538, -42.426, -77.15, -12.087, -67.962, 99.699, 120.725, -2.962, -4.462, 9.699, 6.85, -38.962, -33.462, 40.449, -40.775, 22.288, -42.462, -43.301, 168.975, -81.212, 80.538, -15.926, -176.9, 42.413, 5.288, -46.176, np.nan, np.nan]
        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, random, 3)

    def test_pandas(self):
        res_add = seasonal_decompose(self.data, period=4)
        freq_override_data = self.data.copy()
        freq_override_data.index = pd.date_range(start='1/1/1951', periods=len(freq_override_data), freq=YEAR_END)
        res_add_override = seasonal_decompose(freq_override_data, period=4)
        assert_almost_equal(res_add.seasonal.values.squeeze(), SEASONAL, 2)
        assert_almost_equal(res_add.trend.values.squeeze(), TREND, 2)
        assert_almost_equal(res_add.resid.values.squeeze(), RANDOM, 3)
        assert_almost_equal(res_add_override.seasonal.values.squeeze(), SEASONAL, 2)
        assert_almost_equal(res_add_override.trend.values.squeeze(), TREND, 2)
        assert_almost_equal(res_add_override.resid.values.squeeze(), RANDOM, 3)
        assert_equal(res_add.seasonal.index.values.squeeze(), self.data.index.values)
        res_mult = seasonal_decompose(np.abs(self.data), 'm', period=4)
        res_mult_override = seasonal_decompose(np.abs(freq_override_data), 'm', period=4)
        assert_almost_equal(res_mult.seasonal.values.squeeze(), MULT_SEASONAL, 4)
        assert_almost_equal(res_mult.trend.values.squeeze(), MULT_TREND, 2)
        assert_almost_equal(res_mult.resid.values.squeeze(), MULT_RANDOM, 4)
        assert_almost_equal(res_mult_override.seasonal.values.squeeze(), MULT_SEASONAL, 4)
        assert_almost_equal(res_mult_override.trend.values.squeeze(), MULT_TREND, 2)
        assert_almost_equal(res_mult_override.resid.values.squeeze(), MULT_RANDOM, 4)
        assert_equal(res_mult.seasonal.index.values.squeeze(), self.data.index.values)

    def test_pandas_nofreq(self, reset_randomstate):
        nobs = 100
        dta = pd.Series([x % 3 for x in range(nobs)] + np.random.randn(nobs))
        res_np = seasonal_decompose(dta.values, period=3)
        res = seasonal_decompose(dta, period=3)
        atol = 1e-08
        rtol = 1e-10
        assert_allclose(res.seasonal.values.squeeze(), res_np.seasonal, atol=atol, rtol=rtol)
        assert_allclose(res.trend.values.squeeze(), res_np.trend, atol=atol, rtol=rtol)
        assert_allclose(res.resid.values.squeeze(), res_np.resid, atol=atol, rtol=rtol)

    def test_filt(self):
        filt = np.array([1 / 8.0, 1 / 4.0, 1.0 / 4, 1 / 4.0, 1 / 8.0])
        res_add = seasonal_decompose(self.data.values, filt=filt, period=4)
        assert_almost_equal(res_add.seasonal, SEASONAL, 2)
        assert_almost_equal(res_add.trend, TREND, 2)
        assert_almost_equal(res_add.resid, RANDOM, 3)

    def test_one_sided_moving_average_in_stl_decompose(self):
        res_add = seasonal_decompose(self.data.values, period=4, two_sided=False)
        seasonal = np.array([76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4])
        trend = np.array([np.nan, np.nan, np.nan, np.nan, 159.12, 204.0, 221.25, 245.12, 319.75, 451.5, 561.12, 619.25, 615.62, 548.0, 462.12, 381.12, 316.62, 264.0, 228.38, 210.75, 188.38, 199.0, 207.12, 191.0, 166.88, 72.0, -9.25, -33.12, -36.75, 36.25, 103.0, 131.62])
        resid = np.array([np.nan, np.nan, np.nan, np.nan, 11.112, -57.031, 118.147, 136.272, 332.487, 267.469, 83.272, -77.853, -152.388, -181.031, -152.728, -152.728, -56.388, -115.031, 14.022, -56.353, -33.138, 139.969, -89.728, -40.603, -200.638, -303.031, 46.647, 72.522, 84.987, 234.719, -33.603, 104.772])
        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, resid, 3)
        res_mult = seasonal_decompose(np.abs(self.data.values), 'm', period=4, two_sided=False)
        seasonal = np.array([1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755])
        trend = np.array([np.nan, np.nan, np.nan, np.nan, 171.625, 204.0, 221.25, 245.125, 319.75, 451.5, 561.125, 619.25, 615.625, 548.0, 462.125, 381.125, 316.625, 264.0, 228.375, 210.75, 188.375, 199.0, 207.125, 191.0, 166.875, 107.25, 80.5, 79.125, 78.75, 116.5, 140.0, 157.375])
        resid = np.array([np.nan, np.nan, np.nan, np.nan, 1.2008, 0.752, 1.75, 1.987, 1.9023, 1.1598, 1.6253, 1.169, 0.7319, 0.5398, 0.7261, 0.6837, 0.888, 0.586, 0.9645, 0.7165, 1.0276, 1.3954, 0.0249, 0.7596, 0.215, 0.851, 1.646, 0.2432, 1.3244, 2.0058, 0.5531, 1.7309])
        assert_almost_equal(res_mult.seasonal, seasonal, 4)
        assert_almost_equal(res_mult.trend, trend, 2)
        assert_almost_equal(res_mult.resid, resid, 4)
        res_add = seasonal_decompose(self.data.values[:-1], period=4, two_sided=False)
        seasonal = np.array([81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95])
        trend = [np.nan, np.nan, np.nan, np.nan, 159.12, 204.0, 221.25, 245.12, 319.75, 451.5, 561.12, 619.25, 615.62, 548.0, 462.12, 381.12, 316.62, 264.0, 228.38, 210.75, 188.38, 199.0, 207.12, 191.0, 166.88, 72.0, -9.25, -33.12, -36.75, 36.25, 103.0]
        random = [np.nan, np.nan, np.nan, np.nan, 6.663, -61.48, 113.699, 149.618, 328.038, 263.02, 78.824, -64.507, -156.837, -185.48, -157.176, -139.382, -60.837, -119.48, 9.574, -43.007, -37.587, 135.52, -94.176, -27.257, -205.087, -307.48, 42.199, 85.868, 80.538, 230.27, -38.051]
        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, random, 3)

    def test_2d(self):
        x = np.tile(np.arange(6), (2, 1)).T
        trend = seasonal_decompose(x, period=2).trend
        expected = np.tile(np.arange(6, dtype=float), (2, 1)).T
        expected[0] = expected[-1] = np.nan
        assert_equal(trend, expected)

    def test_interpolate_trend(self):
        x = np.arange(12)
        freq = 4
        trend = seasonal_decompose(x, period=freq).trend
        assert_equal(trend[0], np.nan)
        trend = seasonal_decompose(x, period=freq, extrapolate_trend=5).trend
        assert_almost_equal(trend, x)
        trend = seasonal_decompose(x, period=freq, extrapolate_trend='freq').trend
        assert_almost_equal(trend, x)
        trend = seasonal_decompose(x[:, None], period=freq, extrapolate_trend=5).trend
        assert_almost_equal(trend, x)
        x = np.tile(np.arange(12), (2, 1)).T
        trend = seasonal_decompose(x, period=freq, extrapolate_trend=1).trend
        assert_almost_equal(trend, x)
        trend = seasonal_decompose(x, period=freq, extrapolate_trend='freq').trend
        assert_almost_equal(trend, x)

    def test_raises(self):
        assert_raises(ValueError, seasonal_decompose, self.data.values)
        assert_raises(ValueError, seasonal_decompose, self.data, 'm', period=4)
        x = self.data.astype(float).copy()
        x.iloc[2] = np.nan
        assert_raises(ValueError, seasonal_decompose, x)