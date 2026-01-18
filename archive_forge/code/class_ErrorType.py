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
class ErrorType(PyrexType):
    is_error = 1
    exception_value = '0'
    exception_check = 0
    to_py_function = 'dummy'
    from_py_function = 'dummy'

    def create_to_py_utility_code(self, env):
        return True

    def create_from_py_utility_code(self, env):
        return True

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        return '<error>'

    def same_as_resolved_type(self, other_type):
        return 1

    def error_condition(self, result_code):
        return 'dummy'