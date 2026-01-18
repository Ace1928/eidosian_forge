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
class CEnumDefItemNode(StatNode):
    child_attrs = ['value']

    def analyse_enum_declarations(self, env, enum_entry, incremental_int_value):
        if self.value:
            self.value = self.value.analyse_const_expression(env)
            if not self.value.type.is_int:
                self.value = self.value.coerce_to(PyrexTypes.c_int_type, env)
                self.value = self.value.analyse_const_expression(env)
        if enum_entry.type.is_cpp_enum:
            cname = '%s::%s' % (enum_entry.cname, self.name)
        else:
            cname = self.cname
        self.entry = entry = env.declare_const(self.name, enum_entry.type, self.value, self.pos, cname=cname, visibility=enum_entry.visibility, api=enum_entry.api, create_wrapper=enum_entry.create_wrapper and enum_entry.name is None)
        enum_value = incremental_int_value
        if self.value:
            if self.value.is_literal:
                enum_value = str_to_number(self.value.value)
            elif (self.value.is_name or self.value.is_attribute) and self.value.entry:
                enum_value = self.value.entry.enum_int_value
            else:
                enum_value = None
        if enum_value is not None:
            entry.enum_int_value = enum_value
        enum_entry.enum_values.append(entry)
        if enum_entry.name:
            enum_entry.type.values.append(entry.name)