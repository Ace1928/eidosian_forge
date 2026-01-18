from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class CSimpleBaseTypeNode(CBaseTypeNode):
    child_attrs = []
    arg_name = None
    module_path = []
    is_basic_c_type = False
    complex = False
    is_self_arg = False

    def analyse(self, env, could_be_name=False):
        type = None
        if self.is_basic_c_type:
            type = PyrexTypes.simple_c_type(self.signed, self.longness, self.name)
            if not type:
                error(self.pos, 'Unrecognised type modifier combination')
        elif self.name == 'object' and (not self.module_path):
            type = py_object_type
        elif self.name is None:
            if self.is_self_arg and env.is_c_class_scope:
                type = env.parent_type
            else:
                type = py_object_type
        else:
            scope = env
            if self.module_path:
                for item in self.module_path:
                    entry = scope.lookup(item)
                    if entry is not None and (entry.is_cpp_class or (entry.is_type and entry.type.is_cpp_class)):
                        scope = entry.type.scope
                    elif entry and entry.as_module:
                        scope = entry.as_module
                    else:
                        scope = None
                        break
                if scope is None and len(self.module_path) == 1:
                    from .Builtin import get_known_standard_library_module_scope
                    found_entry = env.lookup(self.module_path[0])
                    if found_entry and found_entry.known_standard_library_import:
                        scope = get_known_standard_library_module_scope(found_entry.known_standard_library_import)
                if scope is None:
                    scope = env.find_imported_module(self.module_path, self.pos)
            if scope:
                if scope.is_c_class_scope:
                    scope = scope.global_scope()
                type = scope.lookup_type(self.name)
                if type is not None:
                    pass
                elif could_be_name:
                    if self.is_self_arg and env.is_c_class_scope:
                        type = env.parent_type
                    else:
                        type = py_object_type
                    self.arg_name = EncodedString(self.name)
                elif self.templates:
                    if self.name not in self.templates:
                        error(self.pos, "'%s' is not a type identifier" % self.name)
                    type = PyrexTypes.TemplatePlaceholderType(self.name)
                else:
                    error(self.pos, "'%s' is not a type identifier" % self.name)
        if type and type.is_fused and env.fused_to_specific:
            type = type.specialize(env.fused_to_specific)
        if self.complex:
            if not type.is_numeric or type.is_complex:
                error(self.pos, 'can only complexify c numeric types')
            type = PyrexTypes.CComplexType(type)
            type.create_declaration_utility_code(env)
        elif type is Builtin.complex_type:
            type = PyrexTypes.c_double_complex_type
            type.create_declaration_utility_code(env)
            self.complex = True
        if not type:
            type = PyrexTypes.error_type
        return type