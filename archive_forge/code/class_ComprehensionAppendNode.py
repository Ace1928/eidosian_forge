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
class ComprehensionAppendNode(Node):
    child_attrs = ['expr']
    target = None
    type = PyrexTypes.c_int_type

    def analyse_expressions(self, env):
        self.expr = self.expr.analyse_expressions(env)
        if not self.expr.type.is_pyobject:
            self.expr = self.expr.coerce_to_pyobject(env)
        return self

    def generate_execution_code(self, code):
        if self.target.type is list_type:
            code.globalstate.use_utility_code(UtilityCode.load_cached('ListCompAppend', 'Optimize.c'))
            function = '__Pyx_ListComp_Append'
        elif self.target.type is set_type:
            function = 'PySet_Add'
        else:
            raise InternalError('Invalid type for comprehension node: %s' % self.target.type)
        self.expr.generate_evaluation_code(code)
        code.putln(code.error_goto_if('%s(%s, (PyObject*)%s)' % (function, self.target.result(), self.expr.result()), self.pos))
        self.expr.generate_disposal_code(code)
        self.expr.free_temps(code)

    def generate_function_definitions(self, env, code):
        self.expr.generate_function_definitions(env, code)

    def annotate(self, code):
        self.expr.annotate(code)