import numpy as np
from numba.core import config
from numba.cuda.testing import CUDATestCase
from numba import cuda
class TestCudaVectorType(CUDATestCase):

    def test_basic(self):
        """Basic test that makes sure that vector type and aliases
        are available within the cuda module from both device and
        simulator mode. This is an important sanity check, since other
        tests below tests the vector type objects programmatically.
        """

        @cuda.jit('void(float64[:])')
        def kernel(arr):
            v1 = cuda.float64x4(1.0, 3.0, 5.0, 7.0)
            v2 = cuda.short2(10, 11)
            arr[0] = v1.x
            arr[1] = v1.y
            arr[2] = v1.z
            arr[3] = v1.w
            arr[4] = v2.x
            arr[5] = v2.y
        res = np.zeros(6, dtype=np.float64)
        kernel[1, 1](res)
        self.assertTrue(np.allclose(res, [1.0, 3.0, 5.0, 7.0, 10, 11]))

    def test_creation_readout(self):
        for vty in vector_types.values():
            with self.subTest(vty=vty):
                arr = np.zeros((vty.num_elements,))
                kernel = make_kernel(vty)
                kernel[1, 1](arr)
                np.testing.assert_almost_equal(arr, np.array(range(vty.num_elements)))

    def test_fancy_creation_readout(self):
        for vty in vector_types.values():
            with self.subTest(vty=vty):
                kernel = make_fancy_creation_kernel(vty)
                expected = np.array([1, 1, 2, 3, 1, 3, 2, 1, 1, 1, 2, 3, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 3, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 3, 4, 1, 2, 1, 4, 1, 2, 3, 1, 1, 1, 3, 4, 1, 2, 1, 4, 1, 2, 3, 1, 1, 1, 1, 4, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 4, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 1, 3, 2, 3, 2, 1, 2, 3, 1, 1, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 1, 1, 2, 3, 1, 1, 4, 2, 3, 1, 4, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 2, 3, 2, 3, 2, 3, 1, 4, 2, 3, 1, 1, 4, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 4])
                arr = np.zeros(expected.shape)
                kernel[1, 1](arr)
                np.testing.assert_almost_equal(arr, expected)

    def test_vector_type_alias(self):
        """Tests that `cuda.<vector_type.alias>` are importable and
        that is the same as `cuda.<vector_type.name>`.

        `test_fancy_creation_readout` only test vector types imported
        with its name. This test makes sure that construction with
        objects imported with alias should work the same.
        """
        for vty in vector_types.values():
            for alias in vty.user_facing_object.aliases:
                with self.subTest(vty=vty.name, alias=alias):
                    self.assertEqual(id(getattr(cuda, vty.name)), id(getattr(cuda, alias)))