import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises
import scipy.fft as fft
from scipy.conftest import (
from scipy._lib._array_api import (
class TestFFTThreadSafe:
    threads = 16
    input_shape = (800, 200)

    def _test_mtsame(self, func, *args, xp=None):

        def worker(args, q):
            q.put(func(*args))
        q = queue.Queue()
        expected = func(*args)
        t = [threading.Thread(target=worker, args=(args, q)) for i in range(self.threads)]
        [x.start() for x in t]
        [x.join() for x in t]
        for i in range(self.threads):
            xp_assert_equal(q.get(timeout=5), expected, err_msg='Function returned wrong value in multithreaded context')

    @array_api_compatible
    def test_fft(self, xp):
        a = xp.ones(self.input_shape, dtype=xp.complex128)
        self._test_mtsame(fft.fft, a, xp=xp)

    @array_api_compatible
    def test_ifft(self, xp):
        a = xp.full(self.input_shape, 1 + 0j)
        self._test_mtsame(fft.ifft, a, xp=xp)

    @array_api_compatible
    def test_rfft(self, xp):
        a = xp.ones(self.input_shape)
        self._test_mtsame(fft.rfft, a, xp=xp)

    @array_api_compatible
    def test_irfft(self, xp):
        a = xp.full(self.input_shape, 1 + 0j)
        self._test_mtsame(fft.irfft, a, xp=xp)

    @array_api_compatible
    def test_hfft(self, xp):
        a = xp.ones(self.input_shape, dtype=xp.complex64)
        self._test_mtsame(fft.hfft, a, xp=xp)

    @array_api_compatible
    def test_ihfft(self, xp):
        a = xp.ones(self.input_shape)
        self._test_mtsame(fft.ihfft, a, xp=xp)