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
class PythonTupleTypeConstructor(BuiltinTypeConstructorObjectType):

    def specialize_here(self, pos, env, template_values=None):
        if template_values and None not in template_values and (not any((v.is_pyobject for v in template_values))):
            entry = env.declare_tuple_type(pos, template_values)
            if entry:
                entry.used = True
                return entry.type
        return super(PythonTupleTypeConstructor, self).specialize_here(pos, env, template_values)