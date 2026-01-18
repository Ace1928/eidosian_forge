import numpy as np
import pandas as pd
def _initialization_heuristic(endog, trend=False, seasonal=False, seasonal_periods=None):
    endog = endog.copy()
    nobs = len(endog)
    if nobs < 10:
        raise ValueError('Cannot use heuristic method with less than 10 observations.')
    initial_seasonal = None
    if seasonal:
        if nobs < 2 * seasonal_periods:
            raise ValueError('Cannot compute initial seasonals using heuristic method with less than two full seasonal cycles in the data.')
        min_obs = 10 + 2 * (seasonal_periods // 2)
        if nobs < min_obs:
            raise ValueError('Cannot use heuristic method to compute initial seasonal and levels with less than 10 + 2 * (seasonal_periods // 2) datapoints.')
        k_cycles = min(5, nobs // seasonal_periods)
        k_cycles = max(k_cycles, int(np.ceil(min_obs / seasonal_periods)))
        series = pd.Series(endog[:seasonal_periods * k_cycles])
        initial_trend = series.rolling(seasonal_periods, center=True).mean()
        if seasonal_periods % 2 == 0:
            initial_trend = initial_trend.shift(-1).rolling(2).mean()
        if seasonal == 'add':
            detrended = series - initial_trend
        elif seasonal == 'mul':
            detrended = series / initial_trend
        tmp = np.zeros(k_cycles * seasonal_periods) * np.nan
        tmp[:len(detrended)] = detrended.values
        initial_seasonal = np.nanmean(tmp.reshape(k_cycles, seasonal_periods).T, axis=1)
        if seasonal == 'add':
            initial_seasonal -= np.mean(initial_seasonal)
        elif seasonal == 'mul':
            initial_seasonal /= np.mean(initial_seasonal)
        endog = initial_trend.dropna().values
    exog = np.c_[np.ones(10), np.arange(10) + 1]
    if endog.ndim == 1:
        endog = np.atleast_2d(endog).T
    beta = np.squeeze(np.linalg.pinv(exog).dot(endog[:10]))
    initial_level = beta[0]
    initial_trend = None
    if trend == 'add':
        initial_trend = beta[1]
    elif trend == 'mul':
        initial_trend = 1 + beta[1] / beta[0]
    return (initial_level, initial_trend, initial_seasonal)