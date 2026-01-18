import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
def _test_array_like_view(self, like_func, view, d_view):
    """
        Tests of device_array_like where the original array is a view - the
        strides should not be equal because a contiguous array is expected.
        """
    nb_like = like_func(d_view)
    self.assertEqual(d_view.shape, nb_like.shape)
    self.assertEqual(d_view.dtype, nb_like.dtype)
    np_like = np.zeros_like(view)
    self.assertEqual(nb_like.strides, np_like.strides)
    self.assertEqual(nb_like.flags['C_CONTIGUOUS'], np_like.flags['C_CONTIGUOUS'])
    self.assertEqual(nb_like.flags['F_CONTIGUOUS'], np_like.flags['F_CONTIGUOUS'])