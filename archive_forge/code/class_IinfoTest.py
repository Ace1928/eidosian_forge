from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
class IinfoTest(parameterized.TestCase):

    def testIinfoInt4(self):
        info = ml_dtypes.iinfo(ml_dtypes.int4)
        self.assertEqual(info.dtype, ml_dtypes.iinfo('int4').dtype)
        self.assertEqual(info.dtype, ml_dtypes.iinfo(np.dtype('int4')).dtype)
        self.assertEqual(info.min, -8)
        self.assertEqual(info.max, 7)
        self.assertEqual(info.dtype, np.dtype(ml_dtypes.int4))
        self.assertEqual(info.bits, 4)
        self.assertEqual(info.kind, 'i')
        self.assertEqual(str(info), 'iinfo(min=-8, max=7, dtype=int4)')

    def testIInfoUint4(self):
        info = ml_dtypes.iinfo(ml_dtypes.uint4)
        self.assertEqual(info.dtype, ml_dtypes.iinfo('uint4').dtype)
        self.assertEqual(info.dtype, ml_dtypes.iinfo(np.dtype('uint4')).dtype)
        self.assertEqual(info.min, 0)
        self.assertEqual(info.max, 15)
        self.assertEqual(info.dtype, np.dtype(ml_dtypes.uint4))
        self.assertEqual(info.bits, 4)
        self.assertEqual(info.kind, 'u')
        self.assertEqual(str(info), 'iinfo(min=0, max=15, dtype=uint4)')

    def testIinfoInt8(self):
        info = ml_dtypes.iinfo(np.int8)
        self.assertEqual(info.min, -128)
        self.assertEqual(info.max, 127)

    def testIinfoNonInteger(self):
        with self.assertRaises(ValueError):
            ml_dtypes.iinfo(np.float32)
        with self.assertRaises(ValueError):
            ml_dtypes.iinfo(np.complex128)
        with self.assertRaises(ValueError):
            ml_dtypes.iinfo(bool)