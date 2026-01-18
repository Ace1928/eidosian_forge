from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
class CFloatType(CNumericType):
    is_float = 1
    to_py_function = 'PyFloat_FromDouble'
    from_py_function = '__pyx_PyFloat_AsDouble'
    exception_value = -1

    def __init__(self, rank, math_h_modifier=''):
        CNumericType.__init__(self, rank, 1)
        self.math_h_modifier = math_h_modifier
        if rank == RANK_FLOAT:
            self.from_py_function = '__pyx_PyFloat_AsFloat'

    def assignable_from_resolved_type(self, src_type):
        return src_type.is_numeric and (not src_type.is_complex) or src_type is error_type

    def invalid_value(self):
        return Naming.PYX_NAN