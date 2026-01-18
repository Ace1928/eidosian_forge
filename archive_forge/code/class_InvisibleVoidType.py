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
class InvisibleVoidType(CVoidType):

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            base_code = '[void]'
        else:
            base_code = public_decl('', dll_linkage)
        return self.base_declaration_code(base_code, entity_code)