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