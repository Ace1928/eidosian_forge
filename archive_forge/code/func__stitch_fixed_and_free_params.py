import numpy as np
from scipy.signal import lfilter
from statsmodels.tools.tools import Bunch
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
def _stitch_fixed_and_free_params(fixed_ar_or_ma_lags, fixed_ar_or_ma_params, free_ar_or_ma_lags, free_ar_or_ma_params, spec_ar_or_ma_lags):
    """
    Stitch together fixed and free params, by the order of lags, for setting
    SARIMAXParams.ma_params or SARIMAXParams.ar_params

    Parameters
    ----------
    fixed_ar_or_ma_lags : list or np.array
    fixed_ar_or_ma_params : list or np.array
        fixed_ar_or_ma_params corresponds with fixed_ar_or_ma_lags
    free_ar_or_ma_lags : list or np.array
    free_ar_or_ma_params : list or np.array
        free_ar_or_ma_params corresponds with free_ar_or_ma_lags
    spec_ar_or_ma_lags : list
        SARIMAXSpecification.ar_lags or SARIMAXSpecification.ma_lags

    Returns
    -------
    list of fixed and free params by the order of lags
    """
    assert len(fixed_ar_or_ma_lags) == len(fixed_ar_or_ma_params)
    assert len(free_ar_or_ma_lags) == len(free_ar_or_ma_params)
    all_lags = np.r_[fixed_ar_or_ma_lags, free_ar_or_ma_lags]
    all_params = np.r_[fixed_ar_or_ma_params, free_ar_or_ma_params]
    assert set(all_lags) == set(spec_ar_or_ma_lags)
    lag_to_param_map = dict(zip(all_lags, all_params))
    all_params_sorted = [lag_to_param_map[lag] for lag in spec_ar_or_ma_lags]
    return all_params_sorted