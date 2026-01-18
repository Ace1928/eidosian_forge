from __future__ import print_function, absolute_import, division
import sys
import gc
import time
import weakref
import threading
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
from .leakcheck import ignores_leakcheck
from .leakcheck import RUNNING_ON_MANYLINUX
class ThreadFunc:
    uss_before = uss_after = 0
    glets = ()
    ITER = 2

    def __call__(self):
        self.uss_before = test.get_process_uss()
        for _ in range(self.ITER):
            self.glets += tuple(run_it())
        for g in self.glets:
            test.assertIn('suspended active', str(g))
        if deallocate_in_thread:
            self.glets = ()
        self.uss_after = test.get_process_uss()