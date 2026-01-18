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
class _FakeStringIO(object):
    """
    A trivial picklable StringIO-alike for Python 2.
    """

    def __init__(self, value):
        self._value = value

    def getvalue(self):
        return self._value