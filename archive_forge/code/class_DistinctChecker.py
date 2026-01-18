import array
from collections import namedtuple
import enum
import mmap
import typing as py_typing
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaValueError, NumbaTypeError
from numba.misc.special import typeof
from numba.core.dispatcher import OmittedArg
from numba._dispatcher import compute_fingerprint
from numba.tests.support import TestCase, skip_unless_cffi, tag
from numba.tests.test_numpy_support import ValueTypingTestBase
from numba.tests.ctypes_usecases import *
from numba.tests.enum_usecases import *
from numba.np import numpy_support
class DistinctChecker(object):

    def __init__(self):
        self._distinct = set()

    def add(self, obj):
        if obj in self._distinct:
            raise AssertionError('%r already in %r' % (obj, self._distinct))
        self._distinct.add(obj)