import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
class Parametrized(tuple):
    """supporting type for TestDictImpl.test_parametrized_types
    needs to be global to be cacheable"""

    def __init__(self, tup):
        assert all((isinstance(v, str) for v in tup))