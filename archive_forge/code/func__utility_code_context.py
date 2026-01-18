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
def _utility_code_context(self):
    return {'type': self.empty_declaration_code(), 'type_name': self.specialization_name(), 'real_type': self.real_type.empty_declaration_code(), 'func_suffix': self.funcsuffix, 'm': self.math_h_modifier, 'is_float': int(self.real_type.is_float), 'is_extern_float_typedef': int(self.real_type.is_float and self.real_type.is_typedef and self.real_type.typedef_is_external)}