from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
class CudaArraySetting(CUDATestCase):
    """
    Most of the slicing logic is tested in the cases above, so these
    tests focus on the setting logic.
    """

    def test_scalar(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        arr[2, 2] = 500
        darr[2, 2] = 500
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_rank(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        arr[2] = 500
        darr[2] = 500
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_broadcast(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        arr[:, 2] = 500
        darr[:, 2] = 500
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_array_assign_column(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        _400 = np.full(shape=7, fill_value=400)
        arr[2] = _400
        darr[2] = _400
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_array_assign_row(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        _400 = np.full(shape=5, fill_value=400)
        arr[:, 2] = _400
        darr[:, 2] = _400
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_array_assign_subarray(self):
        arr = np.arange(5 * 6 * 7).reshape(5, 6, 7)
        darr = cuda.to_device(arr)
        _400 = np.full(shape=(6, 7), fill_value=400)
        arr[2] = _400
        darr[2] = _400
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_array_assign_deep_subarray(self):
        arr = np.arange(5 * 6 * 7 * 8).reshape(5, 6, 7, 8)
        darr = cuda.to_device(arr)
        _400 = np.full(shape=(5, 6, 8), fill_value=400)
        arr[:, :, 2] = _400
        darr[:, :, 2] = _400
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_array_assign_all(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        _400 = np.full(shape=(5, 7), fill_value=400)
        arr[:] = _400
        darr[:] = _400
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_strides(self):
        arr = np.ones(20)
        darr = cuda.to_device(arr)
        arr[::2] = 500
        darr[::2] = 500
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_incompatible_highdim(self):
        darr = cuda.to_device(np.arange(5 * 7))
        with self.assertRaises(ValueError) as e:
            darr[:] = np.ones(shape=(1, 2, 3))
        self.assertIn(member=str(e.exception), container=["Can't assign 3-D array to 1-D self", 'could not broadcast input array from shape (2,3) into shape (35,)'])

    def test_incompatible_shape(self):
        darr = cuda.to_device(np.arange(5))
        with self.assertRaises(ValueError) as e:
            darr[:] = [1, 3]
        self.assertIn(member=str(e.exception), container=["Can't copy sequence with size 2 to array axis 0 with dimension 5", 'could not broadcast input array from shape (2,) into shape (5,)'])

    @skip_on_cudasim('cudasim does not use streams and operates synchronously')
    def test_sync(self):
        darr = cuda.to_device(np.arange(5))
        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
            darr[0] = 10
        mock_sync.assert_called_once()

    @skip_on_cudasim('cudasim does not use streams and operates synchronously')
    def test_no_sync_default_stream(self):
        streams = (cuda.stream(), cuda.default_stream(), cuda.legacy_default_stream(), cuda.per_thread_default_stream())
        for stream in streams:
            darr = cuda.to_device(np.arange(5), stream=stream)
            with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
                darr[0] = 10
            mock_sync.assert_not_called()

    @skip_on_cudasim('cudasim does not use streams and operates synchronously')
    def test_no_sync_supplied_stream(self):
        streams = (cuda.stream(), cuda.default_stream(), cuda.legacy_default_stream(), cuda.per_thread_default_stream())
        for stream in streams:
            darr = cuda.to_device(np.arange(5))
            with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
                darr.setitem(0, 10, stream=stream)
            mock_sync.assert_not_called()

    @unittest.skip('Requires PR #6367')
    def test_issue_6505(self):
        ary = cuda.mapped_array(2, dtype=np.int32)
        ary[:] = 0
        ary_v = ary.view('u1')
        ary_v[1] = 1
        ary_v[5] = 1
        self.assertEqual(sum(ary), 512)