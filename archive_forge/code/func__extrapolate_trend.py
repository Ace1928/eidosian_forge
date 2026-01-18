import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tools.validation import PandasWrapper, array_like
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.stl.mstl import MSTL
from statsmodels.tsa.tsatools import freq_to_period
def _extrapolate_trend(trend, npoints):
    """
    Replace nan values on trend's end-points with least-squares extrapolated
    values with regression considering npoints closest defined points.
    """
    front = next((i for i, vals in enumerate(trend) if not np.any(np.isnan(vals))))
    back = trend.shape[0] - 1 - next((i for i, vals in enumerate(trend[::-1]) if not np.any(np.isnan(vals))))
    front_last = min(front + npoints, back)
    back_first = max(front, back - npoints)
    k, n = np.linalg.lstsq(np.c_[np.arange(front, front_last), np.ones(front_last - front)], trend[front:front_last], rcond=-1)[0]
    extra = (np.arange(0, front) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[:front] = extra
    k, n = np.linalg.lstsq(np.c_[np.arange(back_first, back), np.ones(back - back_first)], trend[back_first:back], rcond=-1)[0]
    extra = (np.arange(back + 1, trend.shape[0]) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[back + 1:] = extra
    return trend