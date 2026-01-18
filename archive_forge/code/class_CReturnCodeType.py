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
class CReturnCodeType(CIntType):
    to_py_function = '__Pyx_Owned_Py_None'
    is_returncode = True
    exception_check = False
    default_format_spec = ''

    def specialization_name(self):
        return super(CReturnCodeType, self).specialization_name() + 'return_code'

    def can_coerce_to_pystring(self, env, format_spec=None):
        return not format_spec

    def convert_to_pystring(self, cvalue, code, format_spec=None):
        return '__Pyx_NewRef(%s)' % code.globalstate.get_py_string_const(StringEncoding.EncodedString('None')).cname