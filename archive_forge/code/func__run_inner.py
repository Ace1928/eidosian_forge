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