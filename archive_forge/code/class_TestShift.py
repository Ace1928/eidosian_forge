from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
class TestShift:

    def test_definition(self):
        for n in [18, 17, 64, 127, 32, 2048, 256]:
            x = arange(n) * 2 * pi / n
            for a in [0.1, 3]:
                assert_array_almost_equal(shift(sin(x), a), direct_shift(sin(x), a))
                assert_array_almost_equal(shift(sin(x), a), sin(x + a))
                assert_array_almost_equal(shift(cos(x), a), cos(x + a))
                assert_array_almost_equal(shift(cos(2 * x) + sin(x), a), cos(2 * (x + a)) + sin(x + a))
                assert_array_almost_equal(shift(exp(sin(x)), a), exp(sin(x + a)))
            assert_array_almost_equal(shift(sin(x), 2 * pi), sin(x))
            assert_array_almost_equal(shift(sin(x), pi), -sin(x))
            assert_array_almost_equal(shift(sin(x), pi / 2), cos(x))