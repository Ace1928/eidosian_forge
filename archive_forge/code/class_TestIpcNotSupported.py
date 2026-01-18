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
@windows_only
@skip_on_cudasim('Ipc not available in CUDASIM')
class TestIpcNotSupported(ContextResettingTestCase):

    def test_unsupported(self):
        arr = np.arange(10, dtype=np.intp)
        devarr = cuda.to_device(arr)
        with self.assertRaises(OSError) as raises:
            devarr.get_ipc_handle()
        errmsg = str(raises.exception)
        self.assertIn('OS does not support CUDA IPC', errmsg)