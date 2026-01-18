import collections
import contextlib
import cProfile
import inspect
import gc
import multiprocessing
import os
import random
import sys
import time
import unittest
import warnings
import zlib
from functools import lru_cache
from io import StringIO
from unittest import result, runner, signals, suite, loader, case
from .loader import TestLoader
from numba.core import config
class RefleakTestResult(runner.TextTestResult):
    warmup = 3
    repetitions = 6

    def _huntLeaks(self, test):
        self.stream.flush()
        repcount = self.repetitions
        nwarmup = self.warmup
        rc_deltas = [0] * (repcount - nwarmup)
        alloc_deltas = [0] * (repcount - nwarmup)
        _int_pool = IntPool()
        for i in range(-200, 200):
            _int_pool[i]
        for i in range(repcount):
            res = result.TestResult()
            test.run(res)
            if not res.wasSuccessful():
                self.failures.extend(res.failures)
                self.errors.extend(res.errors)
                raise AssertionError
            del res
            alloc_after, rc_after = _refleak_cleanup()
            if i >= nwarmup:
                rc_deltas[i - nwarmup] = _int_pool[rc_after - rc_before]
                alloc_deltas[i - nwarmup] = _int_pool[alloc_after - alloc_before]
            alloc_before, rc_before = (alloc_after, rc_after)
        return (rc_deltas, alloc_deltas)

    def addSuccess(self, test):
        try:
            rc_deltas, alloc_deltas = self._huntLeaks(test)
        except AssertionError:
            assert not self.wasSuccessful()
            return

        def check_rc_deltas(deltas):
            return any(deltas)

        def check_alloc_deltas(deltas):
            if 3 * deltas.count(0) < len(deltas):
                return True
            if not set(deltas) <= set((1, 0, -1)):
                return True
            return False
        failed = False
        for deltas, item_name, checker in [(rc_deltas, 'references', check_rc_deltas), (alloc_deltas, 'memory blocks', check_alloc_deltas)]:
            if checker(deltas):
                msg = '%s leaked %s %s, sum=%s' % (test, deltas, item_name, sum(deltas))
                failed = True
                try:
                    raise ReferenceLeakError(msg)
                except Exception:
                    exc_info = sys.exc_info()
                if self.showAll:
                    self.stream.write('%s = %r ' % (item_name, deltas))
                self.addFailure(test, exc_info)
        if not failed:
            super(RefleakTestResult, self).addSuccess(test)