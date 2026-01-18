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
class AsTupleNode(ExprNode):
    subexprs = ['arg']
    is_temp = 1

    def calculate_constant_result(self):
        self.constant_result = tuple(self.arg.constant_result)

    def compile_time_value(self, denv):
        arg = self.arg.compile_time_value(denv)
        try:
            return tuple(arg)
        except Exception as e:
            self.compile_time_value_error(e)

    def analyse_types(self, env):
        self.arg = self.arg.analyse_types(env).coerce_to_pyobject(env)
        if self.arg.type is tuple_type:
            return self.arg.as_none_safe_node("'NoneType' object is not iterable")
        self.type = tuple_type
        return self

    def may_be_none(self):
        return False
    nogil_check = Node.gil_error
    gil_message = 'Constructing Python tuple'

    def generate_result_code(self, code):
        cfunc = '__Pyx_PySequence_Tuple' if self.arg.type in (py_object_type, tuple_type) else 'PySequence_Tuple'
        code.putln('%s = %s(%s); %s' % (self.result(), cfunc, self.arg.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)