import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
class TestCoreContiguous(CUDATestCase):

    def _test_against_array_core(self, view):
        self.assertEqual(devicearray.is_contiguous(view), devicearray.array_core(view).flags['C_CONTIGUOUS'])

    def test_device_array_like_1d(self):
        d_a = cuda.device_array(10, order='C')
        self._test_against_array_core(d_a)

    def test_device_array_like_2d(self):
        d_a = cuda.device_array((10, 12), order='C')
        self._test_against_array_core(d_a)

    def test_device_array_like_2d_transpose(self):
        d_a = cuda.device_array((10, 12), order='C')
        self._test_against_array_core(d_a.T)

    def test_device_array_like_3d(self):
        d_a = cuda.device_array((10, 12, 14), order='C')
        self._test_against_array_core(d_a)

    def test_device_array_like_1d_f(self):
        d_a = cuda.device_array(10, order='F')
        self._test_against_array_core(d_a)

    def test_device_array_like_2d_f(self):
        d_a = cuda.device_array((10, 12), order='F')
        self._test_against_array_core(d_a)

    def test_device_array_like_2d_f_transpose(self):
        d_a = cuda.device_array((10, 12), order='F')
        self._test_against_array_core(d_a.T)

    def test_device_array_like_3d_f(self):
        d_a = cuda.device_array((10, 12, 14), order='F')
        self._test_against_array_core(d_a)

    def test_1d_view(self):
        shape = 10
        view = np.zeros(shape)[::2]
        self._test_against_array_core(view)

    def test_1d_view_f(self):
        shape = 10
        view = np.zeros(shape, order='F')[::2]
        self._test_against_array_core(view)

    def test_2d_view(self):
        shape = (10, 12)
        view = np.zeros(shape)[::2, ::2]
        self._test_against_array_core(view)

    def test_2d_view_f(self):
        shape = (10, 12)
        view = np.zeros(shape, order='F')[::2, ::2]
        self._test_against_array_core(view)