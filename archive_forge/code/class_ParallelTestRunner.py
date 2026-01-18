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
class ParallelTestRunner(runner.TextTestRunner):
    """
    A test runner which delegates the actual running to a pool of child
    processes.
    """
    resultclass = ParallelTestResult
    timeout = _TIMEOUT

    def __init__(self, runner_cls, nprocs, useslice, **kwargs):
        runner.TextTestRunner.__init__(self, **kwargs)
        self.runner_cls = runner_cls
        self.nprocs = nprocs
        self.useslice = parse_slice(useslice)
        self.runner_args = kwargs

    def _run_inner(self, result):
        child_runner = _MinimalRunner(self.runner_cls, self.runner_args)
        chunk_size = 100
        splitted_tests = [self._ptests[i:i + chunk_size] for i in range(0, len(self._ptests), chunk_size)]
        for tests in splitted_tests:
            pool = multiprocessing.Pool(self.nprocs)
            try:
                self._run_parallel_tests(result, pool, child_runner, tests)
            except:
                pool.terminate()
                raise
            else:
                if result.shouldStop:
                    pool.terminate()
                    break
                else:
                    pool.close()
            finally:
                pool.join()
        if not result.shouldStop:
            stests = SerialSuite(self._stests)
            stests.run(result)
            return result

    def _run_parallel_tests(self, result, pool, child_runner, tests):
        remaining_ids = set((t.id() for t in tests))
        tests.sort(key=cuda_sensitive_mtime)
        it = pool.imap_unordered(child_runner, tests)
        while True:
            try:
                child_result = it.__next__(self.timeout)
            except StopIteration:
                return
            except TimeoutError as e:
                msg = "Tests didn't finish before timeout (or crashed):\n%s" % ''.join(('- %r\n' % tid for tid in sorted(remaining_ids)))
                e.args = (msg,) + e.args[1:]
                raise e
            else:
                result.add_results(child_result)
                remaining_ids.discard(child_result.test_id)
                if child_result.shouldStop:
                    result.shouldStop = True
                    return

    def run(self, test):
        self._ptests, self._stests = _split_nonparallel_tests(test, self.useslice)
        print('Parallel: %s. Serial: %s' % (len(self._ptests), len(self._stests)))
        return super(ParallelTestRunner, self).run(self._run_inner)