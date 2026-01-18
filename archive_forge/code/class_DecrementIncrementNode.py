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
class DecrementIncrementNode(CUnopNode):
    is_inc_dec_op = True

    def type_error(self):
        if not self.operand.type.is_error:
            if self.is_prefix:
                error(self.pos, "No match for 'operator%s' (operand type is '%s')" % (self.operator, self.operand.type))
            else:
                error(self.pos, "No 'operator%s(int)' declared for postfix '%s' (operand type is '%s')" % (self.operator, self.operator, self.operand.type))
        self.type = PyrexTypes.error_type

    def analyse_c_operation(self, env):
        if self.operand.type.is_numeric:
            self.type = PyrexTypes.widest_numeric_type(self.operand.type, PyrexTypes.c_int_type)
        elif self.operand.type.is_ptr:
            self.type = self.operand.type
        else:
            self.type_error()

    def calculate_result_code(self):
        if self.is_prefix:
            return '(%s%s)' % (self.operator, self.operand.result())
        else:
            return '(%s%s)' % (self.operand.result(), self.operator)