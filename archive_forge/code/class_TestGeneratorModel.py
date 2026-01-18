import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
class TestGeneratorModel(test_factory()):
    fe_type = types.Generator(gen_func=None, yield_type=types.int32, arg_types=[types.int64, types.float32], state_types=[types.intp, types.intp[::1]], has_finalizer=False)