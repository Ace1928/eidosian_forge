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
def _flatten_suite_inner(test):
    """
    Workhorse for _flatten_suite
    """
    tests = []
    if isinstance(test, (unittest.TestSuite, list, tuple)):
        for x in test:
            tests.extend(_flatten_suite_inner(x))
    else:
        tests.append(test)
    return tests