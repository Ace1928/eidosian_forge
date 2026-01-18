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
class CBIntType(CIntType):
    to_py_function = '__Pyx_PyBool_FromLong'
    from_py_function = '__Pyx_PyObject_IsTrue'
    exception_check = 1
    default_format_spec = ''

    def can_coerce_to_pystring(self, env, format_spec=None):
        return not format_spec or super(CBIntType, self).can_coerce_to_pystring(env, format_spec)

    def convert_to_pystring(self, cvalue, code, format_spec=None):
        if format_spec:
            return super(CBIntType, self).convert_to_pystring(cvalue, code, format_spec)
        utility_code_name = '__Pyx_PyUnicode_FromBInt_' + self.specialization_name()
        to_pyunicode_utility = TempitaUtilityCode.load_cached('CBIntToPyUnicode', 'TypeConversion.c', context={'TRUE_CONST': code.globalstate.get_py_string_const(StringEncoding.EncodedString('True')).cname, 'FALSE_CONST': code.globalstate.get_py_string_const(StringEncoding.EncodedString('False')).cname, 'TO_PY_FUNCTION': utility_code_name})
        code.globalstate.use_utility_code(to_pyunicode_utility)
        return '%s(%s)' % (utility_code_name, cvalue)

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if for_display:
            base_code = 'bool'
        elif pyrex:
            base_code = 'bint'
        else:
            base_code = public_decl('int', dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def specialization_name(self):
        return 'bint'

    def __repr__(self):
        return '<CNumericType bint>'

    def __str__(self):
        return 'bint'

    def py_type_name(self):
        return 'bool'