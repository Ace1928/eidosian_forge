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
class CVoidType(CType):
    is_void = 1
    to_py_function = '__Pyx_void_to_None'

    def __repr__(self):
        return '<CVoidType>'

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            base_code = 'void'
        else:
            base_code = public_decl('void', dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def is_complete(self):
        return 0