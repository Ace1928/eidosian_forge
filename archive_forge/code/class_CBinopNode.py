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
class CBinopNode(BinopNode):

    def analyse_types(self, env):
        node = BinopNode.analyse_types(self, env)
        if node.is_py_operation():
            node.type = PyrexTypes.error_type
        return node

    def py_operation_function(self, code):
        return ''

    def calculate_result_code(self):
        return '(%s %s %s)' % (self.operand1.result(), self.operator, self.operand2.result())

    def compute_c_result_type(self, type1, type2):
        cpp_type = None
        if type1.is_cpp_class or type1.is_ptr:
            cpp_type = type1.find_cpp_operation_type(self.operator, type2)
        if cpp_type is None and (type2.is_cpp_class or type2.is_ptr):
            cpp_type = type2.find_cpp_operation_type(self.operator, type1)
        return cpp_type