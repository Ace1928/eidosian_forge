from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
class CudaArrayStridedSlice(CUDATestCase):

    def test_strided_index_1d(self):
        arr = np.arange(10)
        darr = cuda.to_device(arr)
        for i in range(arr.size):
            np.testing.assert_equal(arr[i::2], darr[i::2].copy_to_host())

    def test_strided_index_2d(self):
        arr = np.arange(6 * 7).reshape(6, 7)
        darr = cuda.to_device(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                np.testing.assert_equal(arr[i::2, j::2], darr[i::2, j::2].copy_to_host())

    def test_strided_index_3d(self):
        arr = np.arange(6 * 7 * 8).reshape(6, 7, 8)
        darr = cuda.to_device(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    np.testing.assert_equal(arr[i::2, j::2, k::2], darr[i::2, j::2, k::2].copy_to_host())