import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@td.skip_if_no('numba')
@pytest.mark.slow
@pytest.mark.filterwarnings('ignore')
class TestTableMethod:

    def test_table_series_valueerror(self):

        def f(x):
            return np.sum(x, axis=0) + 1
        with pytest.raises(ValueError, match="method='table' not applicable for Series objects."):
            Series(range(1)).rolling(1, method='table').apply(f, engine='numba', raw=True)

    def test_table_method_rolling_methods(self, axis, nogil, parallel, nopython, arithmetic_numba_supported_operators, step):
        method, kwargs = arithmetic_numba_supported_operators
        engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        df = DataFrame(np.eye(3))
        roll_table = df.rolling(2, method='table', axis=axis, min_periods=0, step=step)
        if method in ('var', 'std'):
            with pytest.raises(NotImplementedError, match=f'{method} not supported'):
                getattr(roll_table, method)(engine_kwargs=engine_kwargs, engine='numba', **kwargs)
        else:
            roll_single = df.rolling(2, method='single', axis=axis, min_periods=0, step=step)
            result = getattr(roll_table, method)(engine_kwargs=engine_kwargs, engine='numba', **kwargs)
            expected = getattr(roll_single, method)(engine_kwargs=engine_kwargs, engine='numba', **kwargs)
            tm.assert_frame_equal(result, expected)

    def test_table_method_rolling_apply(self, axis, nogil, parallel, nopython, step):
        engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}

        def f(x):
            return np.sum(x, axis=0) + 1
        df = DataFrame(np.eye(3))
        result = df.rolling(2, method='table', axis=axis, min_periods=0, step=step).apply(f, raw=True, engine_kwargs=engine_kwargs, engine='numba')
        expected = df.rolling(2, method='single', axis=axis, min_periods=0, step=step).apply(f, raw=True, engine_kwargs=engine_kwargs, engine='numba')
        tm.assert_frame_equal(result, expected)

    def test_table_method_rolling_weighted_mean(self, step):

        def weighted_mean(x):
            arr = np.ones((1, x.shape[1]))
            arr[:, :2] = (x[:, :2] * x[:, 2]).sum(axis=0) / x[:, 2].sum()
            return arr
        df = DataFrame([[1, 2, 0.6], [2, 3, 0.4], [3, 4, 0.2], [4, 5, 0.7]])
        result = df.rolling(2, method='table', min_periods=0, step=step).apply(weighted_mean, raw=True, engine='numba')
        expected = DataFrame([[1.0, 2.0, 1.0], [1.8, 2.0, 1.0], [3.333333, 2.333333, 1.0], [1.555556, 7, 1.0]])[::step]
        tm.assert_frame_equal(result, expected)

    def test_table_method_expanding_apply(self, axis, nogil, parallel, nopython):
        engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}

        def f(x):
            return np.sum(x, axis=0) + 1
        df = DataFrame(np.eye(3))
        result = df.expanding(method='table', axis=axis).apply(f, raw=True, engine_kwargs=engine_kwargs, engine='numba')
        expected = df.expanding(method='single', axis=axis).apply(f, raw=True, engine_kwargs=engine_kwargs, engine='numba')
        tm.assert_frame_equal(result, expected)

    def test_table_method_expanding_methods(self, axis, nogil, parallel, nopython, arithmetic_numba_supported_operators):
        method, kwargs = arithmetic_numba_supported_operators
        engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        df = DataFrame(np.eye(3))
        expand_table = df.expanding(method='table', axis=axis)
        if method in ('var', 'std'):
            with pytest.raises(NotImplementedError, match=f'{method} not supported'):
                getattr(expand_table, method)(engine_kwargs=engine_kwargs, engine='numba', **kwargs)
        else:
            expand_single = df.expanding(method='single', axis=axis)
            result = getattr(expand_table, method)(engine_kwargs=engine_kwargs, engine='numba', **kwargs)
            expected = getattr(expand_single, method)(engine_kwargs=engine_kwargs, engine='numba', **kwargs)
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('data', [np.eye(3), np.ones((2, 3)), np.ones((3, 2))])
    @pytest.mark.parametrize('method', ['mean', 'sum'])
    def test_table_method_ewm(self, data, method, axis, nogil, parallel, nopython):
        engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        df = DataFrame(data)
        result = getattr(df.ewm(com=1, method='table', axis=axis), method)(engine_kwargs=engine_kwargs, engine='numba')
        expected = getattr(df.ewm(com=1, method='single', axis=axis), method)(engine_kwargs=engine_kwargs, engine='numba')
        tm.assert_frame_equal(result, expected)