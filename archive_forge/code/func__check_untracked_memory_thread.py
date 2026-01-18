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
def _check_untracked_memory_thread(self, deallocate_in_thread=True):
    self._only_test_some_versions()
    EXIT_COUNT = [0]

    def f():
        try:
            greenlet.getcurrent().parent.switch()
        except greenlet.GreenletExit:
            EXIT_COUNT[0] += 1
            raise
        return 1
    ITER = 10000

    def run_it():
        glets = []
        for _ in range(ITER):
            g = greenlet.greenlet(f)
            glets.append(g)
            g.switch()
        return glets
    test = self

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
    uss_before = uss_after = None
    for count in range(self.UNTRACK_ATTEMPTS):
        EXIT_COUNT[0] = 0
        thread_func = ThreadFunc()
        t = threading.Thread(target=thread_func)
        t.start()
        t.join(30)
        self.assertFalse(t.is_alive())
        if uss_before is None:
            uss_before = thread_func.uss_before
        uss_before = max(uss_before, thread_func.uss_before)
        if deallocate_in_thread:
            self.assertEqual(thread_func.glets, ())
            self.assertEqual(EXIT_COUNT[0], ITER * thread_func.ITER)
        del thread_func
        del t
        if not deallocate_in_thread:
            self.assertEqual(EXIT_COUNT[0], 0)
        if deallocate_in_thread:
            self.wait_for_pending_cleanups()
        uss_after = self.get_process_uss()
        if uss_after <= uss_before and count > 1:
            break
    self.wait_for_pending_cleanups()
    uss_after = self.get_process_uss()
    self.assertLessEqual(uss_after, uss_before, 'after attempts %d' % (count,))