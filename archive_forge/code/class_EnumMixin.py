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
class EnumMixin(object):
    """
    Common implementation details for C and C++ enums.
    """

    def create_enum_to_py_utility_code(self, env):
        from .UtilityCode import CythonUtilityCode
        self.to_py_function = '__Pyx_Enum_%s_to_py' % type_identifier(self)
        if self.entry.scope != env.global_scope():
            module_name = self.entry.scope.qualified_name
        else:
            module_name = None
        directives = CythonUtilityCode.filter_inherited_directives(env.global_scope().directives)
        if any((value_entry.enum_int_value is None for value_entry in self.entry.enum_values)):
            directives['optimize.use_switch'] = False
        if self.is_cpp_enum:
            underlying_type_str = self.underlying_type.empty_declaration_code()
        else:
            underlying_type_str = 'int'
        env.use_utility_code(CythonUtilityCode.load('EnumTypeToPy', 'CpdefEnums.pyx', context={'funcname': self.to_py_function, 'name': self.name, 'items': tuple(self.values), 'underlying_type': underlying_type_str, 'module_name': module_name, 'is_flag': not self.is_cpp_enum}, outer_module_scope=self.entry.scope, compiler_directives=directives))