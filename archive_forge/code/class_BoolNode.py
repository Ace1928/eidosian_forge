from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class BoolNode(ConstNode):
    type = PyrexTypes.c_bint_type

    def calculate_constant_result(self):
        self.constant_result = self.value

    def compile_time_value(self, denv):
        return self.value

    def calculate_result_code(self):
        if self.type.is_pyobject:
            return 'Py_True' if self.value else 'Py_False'
        else:
            return str(int(self.value))

    def coerce_to(self, dst_type, env):
        if dst_type == self.type:
            return self
        if dst_type is py_object_type and self.type is Builtin.bool_type:
            return self
        if dst_type.is_pyobject and self.type.is_int:
            return BoolNode(self.pos, value=self.value, constant_result=self.constant_result, type=Builtin.bool_type)
        if dst_type.is_int and self.type.is_pyobject:
            return BoolNode(self.pos, value=self.value, constant_result=self.constant_result, type=PyrexTypes.c_bint_type)
        return ConstNode.coerce_to(self, dst_type, env)