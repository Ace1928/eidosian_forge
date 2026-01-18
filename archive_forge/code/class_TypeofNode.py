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
class TypeofNode(ExprNode):
    literal = None
    type = py_object_type
    subexprs = ['literal']

    def analyse_types(self, env):
        self.operand = self.operand.analyse_types(env)
        value = StringEncoding.EncodedString(str(self.operand.type))
        literal = StringNode(self.pos, value=value)
        literal = literal.analyse_types(env)
        self.literal = literal.coerce_to_pyobject(env)
        return self

    def analyse_as_type(self, env):
        self.operand = self.operand.analyse_types(env)
        return self.operand.type

    def may_be_none(self):
        return False

    def generate_evaluation_code(self, code):
        self.literal.generate_evaluation_code(code)

    def calculate_result_code(self):
        return self.literal.calculate_result_code()