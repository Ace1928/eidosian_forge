import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy
@structref.register
class MyStructType(types.StructRef):

    def preprocess_fields(self, fields):
        return tuple(((name, types.unliteral(typ)) for name, typ in fields))