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
class CPyUCS4IntType(CIntType):
    is_unicode_char = True
    to_py_function = '__Pyx_PyUnicode_FromOrdinal'
    from_py_function = '__Pyx_PyObject_AsPy_UCS4'

    def can_coerce_to_pystring(self, env, format_spec=None):
        return False

    def create_from_py_utility_code(self, env):
        env.use_utility_code(UtilityCode.load_cached('ObjectAsUCS4', 'TypeConversion.c'))
        return True

    def sign_and_name(self):
        return 'Py_UCS4'