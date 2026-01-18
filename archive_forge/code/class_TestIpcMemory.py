import multiprocessing as mp
import itertools
import traceback
import pickle
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import (skip_on_arm, skip_on_cudasim,
from numba.tests.support import linux_only, windows_only
import unittest
@linux_only
@skip_under_cuda_memcheck('Hangs cuda-memcheck')
@skip_on_cudasim('Ipc not available in CUDASIM')
@skip_on_arm('CUDA IPC not supported on ARM in Numba')
class TestIpcMemory(ContextResettingTestCase):

    def test_ipc_handle(self):
        arr = np.arange(10, dtype=np.intp)
        devarr = cuda.to_device(arr)
        ctx = cuda.current_context()
        ipch = ctx.get_ipc_handle(devarr.gpu_data)
        if driver.USE_NV_BINDING:
            handle_bytes = ipch.handle.reserved
        else:
            handle_bytes = bytes(ipch.handle)
        size = ipch.size
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        args = (handle_bytes, size, result_queue)
        proc = ctx.Process(target=base_ipc_handle_test, args=args)
        proc.start()
        succ, out = result_queue.get()
        if not succ:
            self.fail(out)
        else:
            np.testing.assert_equal(arr, out)
        proc.join(3)

    def variants(self):
        indices = (None, slice(3, None), slice(3, 8), slice(None, 8))
        foreigns = (False, True)
        return itertools.product(indices, foreigns)

    def check_ipc_handle_serialization(self, index_arg=None, foreign=False):
        arr = np.arange(10, dtype=np.intp)
        devarr = cuda.to_device(arr)
        if index_arg is not None:
            devarr = devarr[index_arg]
        if foreign:
            devarr = cuda.as_cuda_array(ForeignArray(devarr))
        expect = devarr.copy_to_host()
        ctx = cuda.current_context()
        ipch = ctx.get_ipc_handle(devarr.gpu_data)
        buf = pickle.dumps(ipch)
        ipch_recon = pickle.loads(buf)
        self.assertIs(ipch_recon.base, None)
        self.assertEqual(ipch_recon.size, ipch.size)
        if driver.USE_NV_BINDING:
            self.assertEqual(ipch_recon.handle.reserved, ipch.handle.reserved)
        else:
            self.assertEqual(tuple(ipch_recon.handle), tuple(ipch.handle))
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        args = (ipch, result_queue)
        proc = ctx.Process(target=serialize_ipc_handle_test, args=args)
        proc.start()
        succ, out = result_queue.get()
        if not succ:
            self.fail(out)
        else:
            np.testing.assert_equal(expect, out)
        proc.join(3)

    def test_ipc_handle_serialization(self):
        for index, foreign in self.variants():
            with self.subTest(index=index, foreign=foreign):
                self.check_ipc_handle_serialization(index, foreign)

    def check_ipc_array(self, index_arg=None, foreign=False):
        arr = np.arange(10, dtype=np.intp)
        devarr = cuda.to_device(arr)
        if index_arg is not None:
            devarr = devarr[index_arg]
        if foreign:
            devarr = cuda.as_cuda_array(ForeignArray(devarr))
        expect = devarr.copy_to_host()
        ipch = devarr.get_ipc_handle()
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        args = (ipch, result_queue)
        proc = ctx.Process(target=ipc_array_test, args=args)
        proc.start()
        succ, out = result_queue.get()
        if not succ:
            self.fail(out)
        else:
            np.testing.assert_equal(expect, out)
        proc.join(3)

    def test_ipc_array(self):
        for index, foreign in self.variants():
            with self.subTest(index=index, foreign=foreign):
                self.check_ipc_array(index, foreign)