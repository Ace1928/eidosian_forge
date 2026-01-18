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
class SizeofVarNode(SizeofNode):
    subexprs = ['operand']

    def analyse_types(self, env):
        operand_as_type = self.operand.analyse_as_type(env)
        if operand_as_type:
            self.arg_type = operand_as_type
            if self.arg_type.is_fused:
                try:
                    self.arg_type = self.arg_type.specialize(env.fused_to_specific)
                except CannotSpecialize:
                    error(self.operand.pos, 'Type cannot be specialized since it is not a fused argument to this function')
            self.__class__ = SizeofTypeNode
            self.check_type()
        else:
            self.operand = self.operand.analyse_types(env)
        return self

    def calculate_result_code(self):
        return '(sizeof(%s))' % self.operand.result()

    def generate_result_code(self, code):
        pass