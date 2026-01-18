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
class TestCorrelate:

    def test_invalid_shapes(self):
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))
        assert_raises(ValueError, correlate, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, correlate, *(b, a), **{'mode': 'valid'})

    def test_invalid_params(self):
        a = [3, 4, 5]
        b = [1, 2, 3]
        assert_raises(ValueError, correlate, a, b, mode='spam')
        assert_raises(ValueError, correlate, a, b, mode='eggs', method='fft')
        assert_raises(ValueError, correlate, a, b, mode='ham', method='direct')
        assert_raises(ValueError, correlate, a, b, mode='full', method='bacon')
        assert_raises(ValueError, correlate, a, b, mode='same', method='bacon')

    def test_mismatched_dims(self):
        assert_raises(ValueError, correlate, [1], 2, method='direct')
        assert_raises(ValueError, correlate, 1, [2], method='direct')
        assert_raises(ValueError, correlate, [1], 2, method='fft')
        assert_raises(ValueError, correlate, 1, [2], method='fft')
        assert_raises(ValueError, correlate, [1], [[2]])
        assert_raises(ValueError, correlate, [3], 2)

    def test_numpy_fastpath(self):
        a = [1, 2, 3]
        b = [4, 5]
        assert_allclose(correlate(a, b, mode='same'), [5, 14, 23])
        a = [1, 2, 3]
        b = [4, 5, 6]
        assert_allclose(correlate(a, b, mode='same'), [17, 32, 23])
        assert_allclose(correlate(a, b, mode='full'), [6, 17, 32, 23, 12])
        assert_allclose(correlate(a, b, mode='valid'), [32])