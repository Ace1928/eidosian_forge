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
def _choose_tagged_tests(tests, tags, mode='include'):
    """
    Select tests that are tagged/not tagged with at least one of the given tags.
    Set mode to 'include' to include the tests with tags, or 'exclude' to
    exclude the tests with the tags.
    """
    selected = []
    tags = set(tags)
    for test in _flatten_suite(tests):
        assert isinstance(test, unittest.TestCase)
        func = getattr(test, test._testMethodName)
        try:
            func = func.im_func
        except AttributeError:
            pass
        found_tags = getattr(func, 'tags', None)
        if mode == 'include':
            if found_tags is not None and found_tags & tags:
                selected.append(test)
        elif mode == 'exclude':
            if found_tags is None or not found_tags & tags:
                selected.append(test)
        else:
            raise ValueError("Invalid 'mode' supplied: %s." % mode)
    return unittest.TestSuite(selected)