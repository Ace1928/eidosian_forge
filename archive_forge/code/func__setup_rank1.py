import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
def _setup_rank1(self, dt, mode):
    np.random.seed(9)
    a = np.random.randn(10).astype(dt)
    a += 1j * np.random.randn(10).astype(dt)
    b = np.random.randn(8).astype(dt)
    b += 1j * np.random.randn(8).astype(dt)
    y_r = (correlate(a.real, b.real, mode=mode) + correlate(a.imag, b.imag, mode=mode)).astype(dt)
    y_r += 1j * (-correlate(a.real, b.imag, mode=mode) + correlate(a.imag, b.real, mode=mode))
    return (a, b, y_r)