from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
@skip_on_cudasim('Overloading not supported in cudasim')
class TestOverload(CUDATestCase):

    def check_overload(self, kernel, expected):
        x = np.ones(1, dtype=np.int32)
        cuda.jit(kernel)[1, 1](x)
        self.assertEqual(x[0], expected)

    def check_overload_cpu(self, kernel, expected):
        x = np.ones(1, dtype=np.int32)
        njit(kernel)(x)
        self.assertEqual(x[0], expected)

    def test_generic(self):

        def kernel(x):
            generic_func_1(x)
        expected = GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda(self):

        def kernel(x):
            cuda_func_1(x)
        expected = CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_generic_and_cuda(self):

        def kernel(x):
            generic_func_1(x)
            cuda_func_1(x)
        expected = GENERIC_FUNCTION_1 * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_call_two_generic_calls(self):

        def kernel(x):
            generic_func_1(x)
            generic_func_2(x)
        expected = GENERIC_FUNCTION_1 * GENERIC_FUNCTION_2
        self.check_overload(kernel, expected)

    def test_call_two_cuda_calls(self):

        def kernel(x):
            cuda_func_1(x)
            cuda_func_2(x)
        expected = CUDA_FUNCTION_1 * CUDA_FUNCTION_2
        self.check_overload(kernel, expected)

    def test_generic_calls_generic(self):

        def kernel(x):
            generic_calls_generic(x)
        expected = GENERIC_CALLS_GENERIC * GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_generic_calls_cuda(self):

        def kernel(x):
            generic_calls_cuda(x)
        expected = GENERIC_CALLS_CUDA * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda_calls_generic(self):

        def kernel(x):
            cuda_calls_generic(x)
        expected = CUDA_CALLS_GENERIC * GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda_calls_cuda(self):

        def kernel(x):
            cuda_calls_cuda(x)
        expected = CUDA_CALLS_CUDA * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_call_target_overloaded(self):

        def kernel(x):
            target_overloaded(x)
        expected = CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_generic_calls_target_overloaded(self):

        def kernel(x):
            generic_calls_target_overloaded(x)
        expected = GENERIC_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_cuda_calls_target_overloaded(self):

        def kernel(x):
            cuda_calls_target_overloaded(x)
        expected = CUDA_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_target_overloaded_calls_target_overloaded(self):

        def kernel(x):
            target_overloaded_calls_target_overloaded(x)
        expected = CUDA_TARGET_OL_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)
        expected = GENERIC_TARGET_OL_CALLS_TARGET_OL * GENERIC_TARGET_OL
        self.check_overload_cpu(kernel, expected)