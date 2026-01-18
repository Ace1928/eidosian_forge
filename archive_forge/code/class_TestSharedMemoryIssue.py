from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
class TestSharedMemoryIssue(CUDATestCase):

    def test_issue_953_sm_linkage_conflict(self):

        @cuda.jit(device=True)
        def inner():
            inner_arr = cuda.shared.array(1, dtype=int32)

        @cuda.jit
        def outer():
            outer_arr = cuda.shared.array(1, dtype=int32)
            inner()
        outer[1, 1]()

    def _check_shared_array_size(self, shape, expected):

        @cuda.jit
        def s(a):
            arr = cuda.shared.array(shape, dtype=int32)
            a[0] = arr.size
        result = np.zeros(1, dtype=np.int32)
        s[1, 1](result)
        self.assertEqual(result[0], expected)

    def test_issue_1051_shared_size_broken_1d(self):
        self._check_shared_array_size(2, 2)

    def test_issue_1051_shared_size_broken_2d(self):
        self._check_shared_array_size((2, 3), 6)

    def test_issue_1051_shared_size_broken_3d(self):
        self._check_shared_array_size((2, 3, 4), 24)

    def _check_shared_array_size_fp16(self, shape, expected, ty):

        @cuda.jit
        def s(a):
            arr = cuda.shared.array(shape, dtype=ty)
            a[0] = arr.size
        result = np.zeros(1, dtype=np.float16)
        s[1, 1](result)
        self.assertEqual(result[0], expected)

    def test_issue_fp16_support(self):
        self._check_shared_array_size_fp16(2, 2, types.float16)
        self._check_shared_array_size_fp16(2, 2, np.float16)

    def test_issue_2393(self):
        """
        Test issue of warp misalign address due to nvvm not knowing the
        alignment(? but it should have taken the natural alignment of the type)
        """
        num_weights = 2
        num_blocks = 48
        examples_per_block = 4
        threads_per_block = 1

        @cuda.jit
        def costs_func(d_block_costs):
            s_features = cuda.shared.array((examples_per_block, num_weights), float64)
            s_initialcost = cuda.shared.array(7, float64)
            threadIdx = cuda.threadIdx.x
            prediction = 0
            for j in range(num_weights):
                prediction += s_features[threadIdx, j]
            d_block_costs[0] = s_initialcost[0] + prediction
        block_costs = np.zeros(num_blocks, dtype=np.float64)
        d_block_costs = cuda.to_device(block_costs)
        costs_func[num_blocks, threads_per_block](d_block_costs)
        cuda.synchronize()