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
class CppRvalueReferenceType(CReferenceBaseType):
    is_rvalue_reference = 1

    def __str__(self):
        return '%s &&' % self.ref_base_type

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        return self.ref_base_type.declaration_code('&&%s' % entity_code, for_display, dll_linkage, pyrex)