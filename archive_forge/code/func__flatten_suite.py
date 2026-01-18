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
def _flatten_suite(test):
    """
    Expand nested suite into list of test cases.
    """
    tests = _flatten_suite_inner(test)
    generated = set()
    for t in tests:
        for g in _GENERATED:
            if g in str(t):
                generated.add(t)
    normal = set(tests) - generated

    def key(x):
        return (x.__module__, type(x).__name__, x._testMethodName)
    tests = sorted(normal, key=key)
    tests.extend(sorted(list(generated), key=key))
    return tests