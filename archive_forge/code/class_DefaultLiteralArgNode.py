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
class DefaultLiteralArgNode(ExprNode):
    subexprs = []
    is_literal = True
    is_temp = False

    def __init__(self, pos, arg):
        super(DefaultLiteralArgNode, self).__init__(pos)
        self.arg = arg
        self.constant_result = arg.constant_result
        self.type = self.arg.type
        self.evaluated = False

    def analyse_types(self, env):
        return self

    def generate_result_code(self, code):
        pass

    def generate_evaluation_code(self, code):
        if not self.evaluated:
            self.arg.generate_evaluation_code(code)
            self.evaluated = True

    def result(self):
        return self.type.cast_code(self.arg.result())