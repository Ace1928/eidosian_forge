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
class CoercionNode(ExprNode):
    subexprs = ['arg']
    constant_result = not_a_constant

    def __init__(self, arg):
        super(CoercionNode, self).__init__(arg.pos)
        self.arg = arg
        if debug_coercion:
            print('%s Coercing %s' % (self, self.arg))

    def calculate_constant_result(self):
        pass

    def annotate(self, code):
        self.arg.annotate(code)
        if self.arg.type != self.type:
            file, line, col = self.pos
            code.annotate((file, line, col - 1), AnnotationItem(style='coerce', tag='coerce', text='[%s] to [%s]' % (self.arg.type, self.type)))

    def analyse_types(self, env):
        return self