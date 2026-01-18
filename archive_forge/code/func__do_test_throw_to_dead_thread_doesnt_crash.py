from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import sys
import time
import threading
from abc import ABCMeta, abstractmethod
import greenlet
from greenlet import greenlet as RawGreenlet
from . import TestCase
from .leakcheck import fails_leakcheck
@fails_leakcheck
def _do_test_throw_to_dead_thread_doesnt_crash(self, wait_for_cleanup=False):
    result = []

    def worker():
        greenlet.getcurrent().parent.switch()

    def creator():
        g = RawGreenlet(worker)
        g.switch()
        result.append(g)
        if wait_for_cleanup:
            g.switch()
            greenlet.getcurrent()
    t = threading.Thread(target=creator)
    t.start()
    t.join(10)
    del t
    if wait_for_cleanup:
        self.wait_for_pending_cleanups()
    with self.assertRaises(greenlet.error) as exc:
        result[0].throw(SomeError)
    if not wait_for_cleanup:
        self.assertIn(str(exc.exception), ['cannot switch to a different thread (which happens to have exited)', 'cannot switch to a different thread'])
    else:
        self.assertEqual(str(exc.exception), 'cannot switch to a different thread (which happens to have exited)')
    if hasattr(result[0].gr_frame, 'clear'):
        with self.assertRaises(RuntimeError):
            result[0].gr_frame.clear()
    if not wait_for_cleanup:
        result[0].gr_frame.f_locals.clear()
    else:
        self.assertIsNone(result[0].gr_frame)
    del creator
    worker = None
    del result[:]
    self.expect_greenlet_leak = True