import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
@staticmethod
def _check_nan_policy(f, xdata_with_nan, xdata_without_nan, ydata_with_nan, ydata_without_nan, method):
    kwargs = {'f': f, 'xdata': xdata_with_nan, 'ydata': ydata_with_nan, 'method': method, 'check_finite': False}
    error_msg = "`nan_policy='propagate'` is not supported by this function."
    with assert_raises(ValueError, match=error_msg):
        curve_fit(**kwargs, nan_policy='propagate', maxfev=2000)
    with assert_raises(ValueError, match='The input contains nan'):
        curve_fit(**kwargs, nan_policy='raise')
    result_with_nan, _ = curve_fit(**kwargs, nan_policy='omit')
    kwargs['xdata'] = xdata_without_nan
    kwargs['ydata'] = ydata_without_nan
    result_without_nan, _ = curve_fit(**kwargs)
    assert_allclose(result_with_nan, result_without_nan)
    error_msg = "nan_policy must be one of {'None', 'raise', 'omit'}"
    with assert_raises(ValueError, match=error_msg):
        curve_fit(**kwargs, nan_policy='hi')