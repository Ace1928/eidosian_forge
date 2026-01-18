import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
class TestFirwin:

    def check_response(self, h, expected_response, tol=0.05):
        N = len(h)
        alpha = 0.5 * (N - 1)
        m = np.arange(0, N) - alpha
        for freq, expected in expected_response:
            actual = abs(np.sum(h * np.exp(-1j * np.pi * m * freq)))
            mse = abs(actual - expected) ** 2
            assert_(mse < tol, f'response not as expected, mse={mse:g} > {tol:g}')

    def test_response(self):
        N = 51
        f = 0.5
        h = firwin(N, f)
        self.check_response(h, [(0.25, 1), (0.75, 0)])
        h = firwin(N + 1, f, window='nuttall')
        self.check_response(h, [(0.25, 1), (0.75, 0)])
        h = firwin(N + 2, f, pass_zero=False)
        self.check_response(h, [(0.25, 0), (0.75, 1)])
        f1, f2, f3, f4 = (0.2, 0.4, 0.6, 0.8)
        h = firwin(N + 3, [f1, f2], pass_zero=False)
        self.check_response(h, [(0.1, 0), (0.3, 1), (0.5, 0)])
        h = firwin(N + 4, [f1, f2])
        self.check_response(h, [(0.1, 1), (0.3, 0), (0.5, 1)])
        h = firwin(N + 5, [f1, f2, f3, f4], pass_zero=False, scale=False)
        self.check_response(h, [(0.1, 0), (0.3, 1), (0.5, 0), (0.7, 1), (0.9, 0)])
        h = firwin(N + 6, [f1, f2, f3, f4])
        self.check_response(h, [(0.1, 1), (0.3, 0), (0.5, 1), (0.7, 0), (0.9, 1)])
        h = firwin(N + 7, 0.1, width=0.03)
        self.check_response(h, [(0.05, 1), (0.75, 0)])
        h = firwin(N + 8, 0.1, pass_zero=False)
        self.check_response(h, [(0.05, 0), (0.75, 1)])

    def mse(self, h, bands):
        """Compute mean squared error versus ideal response across frequency
        band.
          h -- coefficients
          bands -- list of (left, right) tuples relative to 1==Nyquist of
            passbands
        """
        w, H = freqz(h, worN=1024)
        f = w / np.pi
        passIndicator = np.zeros(len(w), bool)
        for left, right in bands:
            passIndicator |= (f >= left) & (f < right)
        Hideal = np.where(passIndicator, 1, 0)
        mse = np.mean(abs(abs(H) - Hideal) ** 2)
        return mse

    def test_scaling(self):
        """
        For one lowpass, bandpass, and highpass example filter, this test
        checks two things:
          - the mean squared error over the frequency domain of the unscaled
            filter is smaller than the scaled filter (true for rectangular
            window)
          - the response of the scaled filter is exactly unity at the center
            of the first passband
        """
        N = 11
        cases = [([0.5], True, (0, 1)), ([0.2, 0.6], False, (0.4, 1)), ([0.5], False, (1, 1))]
        for cutoff, pass_zero, expected_response in cases:
            h = firwin(N, cutoff, scale=False, pass_zero=pass_zero, window='ones')
            hs = firwin(N, cutoff, scale=True, pass_zero=pass_zero, window='ones')
            if len(cutoff) == 1:
                if pass_zero:
                    cutoff = [0] + cutoff
                else:
                    cutoff = cutoff + [1]
            assert_(self.mse(h, [cutoff]) < self.mse(hs, [cutoff]), 'least squares violation')
            self.check_response(hs, [expected_response], 1e-12)