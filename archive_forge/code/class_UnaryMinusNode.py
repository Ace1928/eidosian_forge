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
class UnaryMinusNode(UnopNode):
    operator = '-'

    def analyse_c_operation(self, env):
        if self.operand.type.is_numeric:
            self.type = PyrexTypes.widest_numeric_type(self.operand.type, PyrexTypes.c_int_type)
        elif self.operand.type.is_enum:
            self.type = PyrexTypes.c_int_type
        else:
            self.type_error()
        if self.type.is_complex:
            self.infix = False

    def py_operation_function(self, code):
        return 'PyNumber_Negative'

    def calculate_result_code(self):
        if self.infix:
            return '(-%s)' % self.operand.result()
        else:
            return '%s(%s)' % (self.operand.type.unary_op('-'), self.operand.result())

    def get_constant_c_result_code(self):
        value = self.operand.get_constant_c_result_code()
        if value:
            return '(-%s)' % value