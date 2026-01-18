import sys
import os
import multiprocessing as mp
import warnings
from numba.core.config import IS_WIN32, IS_OSX
from numba.core.errors import NumbaWarning
from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import (
from numba.cuda.cuda_paths import (
class LibraryLookupBase(SerialMixin, unittest.TestCase):

    def setUp(self):
        ctx = mp.get_context('spawn')
        qrecv = ctx.Queue()
        qsend = ctx.Queue()
        self.qsend = qsend
        self.qrecv = qrecv
        self.child_process = ctx.Process(target=check_lib_lookup, args=(qrecv, qsend), daemon=True)
        self.child_process.start()

    def tearDown(self):
        self.qsend.put(self.do_terminate)
        self.child_process.join(3)
        self.assertIsNotNone(self.child_process)

    def remote_do(self, action):
        self.qsend.put(action)
        out = self.qrecv.get()
        self.assertNotIsInstance(out, BaseException)
        return out

    @staticmethod
    def do_terminate():
        return (False, None)