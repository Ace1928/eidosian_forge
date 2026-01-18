import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
@skip_on_cudasim('Device Record API unsupported in the simulator')
class TestCudaDeviceRecord(CUDATestCase):
    """
    Tests the DeviceRecord class with np.void host types.
    """

    def setUp(self):
        super().setUp()
        self._create_data(np.zeros)

    def _create_data(self, array_ctor):
        self.dtype = np.dtype([('a', np.int32), ('b', np.float32)], align=True)
        self.hostz = array_ctor(1, self.dtype)[0]
        self.hostnz = array_ctor(1, self.dtype)[0]
        self.hostnz['a'] = 10
        self.hostnz['b'] = 11.0

    def _check_device_record(self, reference, rec):
        self.assertEqual(rec.shape, tuple())
        self.assertEqual(rec.strides, tuple())
        self.assertEqual(rec.dtype, reference.dtype)
        self.assertEqual(rec.alloc_size, reference.dtype.itemsize)
        self.assertIsNotNone(rec.gpu_data)
        self.assertNotEqual(rec.device_ctypes_pointer, ctypes.c_void_p(0))
        numba_type = numpy_support.from_dtype(reference.dtype)
        self.assertEqual(rec._numba_type_, numba_type)

    def test_device_record_interface(self):
        hostrec = self.hostz.copy()
        devrec = DeviceRecord(self.dtype)
        self._check_device_record(hostrec, devrec)

    def test_device_record_copy(self):
        hostrec = self.hostz.copy()
        devrec = DeviceRecord(self.dtype)
        devrec.copy_to_device(hostrec)
        hostrec2 = self.hostnz.copy()
        devrec.copy_to_host(hostrec2)
        np.testing.assert_equal(self.hostz, hostrec2)
        hostrec3 = self.hostnz.copy()
        devrec.copy_to_device(hostrec3)
        hostrec4 = self.hostz.copy()
        devrec.copy_to_host(hostrec4)
        np.testing.assert_equal(hostrec4, self.hostnz)

    def test_from_record_like(self):
        hostrec = self.hostz.copy()
        devrec = from_record_like(hostrec)
        self._check_device_record(hostrec, devrec)
        devrec2 = from_record_like(devrec)
        self._check_device_record(devrec, devrec2)
        self.assertNotEqual(devrec.gpu_data, devrec2.gpu_data)

    def test_auto_device(self):
        hostrec = self.hostnz.copy()
        devrec, new_gpu_obj = auto_device(hostrec)
        self._check_device_record(hostrec, devrec)
        self.assertTrue(new_gpu_obj)
        hostrec2 = self.hostz.copy()
        devrec.copy_to_host(hostrec2)
        np.testing.assert_equal(hostrec2, hostrec)