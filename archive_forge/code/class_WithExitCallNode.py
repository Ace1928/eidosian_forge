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
class WithExitCallNode(ExprNode):
    subexprs = ['args', 'await_expr']
    test_if_run = True
    await_expr = None

    def analyse_types(self, env):
        self.args = self.args.analyse_types(env)
        if self.await_expr:
            self.await_expr = self.await_expr.analyse_types(env)
        self.type = PyrexTypes.c_bint_type
        self.is_temp = True
        return self

    def generate_evaluation_code(self, code):
        if self.test_if_run:
            code.putln('if (%s) {' % self.with_stat.exit_var)
        self.args.generate_evaluation_code(code)
        result_var = code.funcstate.allocate_temp(py_object_type, manage_ref=False)
        code.mark_pos(self.pos)
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCall', 'ObjectHandling.c'))
        code.putln('%s = __Pyx_PyObject_Call(%s, %s, NULL);' % (result_var, self.with_stat.exit_var, self.args.result()))
        code.put_decref_clear(self.with_stat.exit_var, type=py_object_type)
        self.args.generate_disposal_code(code)
        self.args.free_temps(code)
        code.putln(code.error_goto_if_null(result_var, self.pos))
        code.put_gotref(result_var, py_object_type)
        if self.await_expr:
            self.await_expr.generate_evaluation_code(code, source_cname=result_var, decref_source=True)
            code.putln('%s = %s;' % (result_var, self.await_expr.py_result()))
            self.await_expr.generate_post_assignment_code(code)
            self.await_expr.free_temps(code)
        if self.result_is_used:
            self.allocate_temp_result(code)
            code.putln('%s = __Pyx_PyObject_IsTrue(%s);' % (self.result(), result_var))
        code.put_decref_clear(result_var, type=py_object_type)
        if self.result_is_used:
            code.put_error_if_neg(self.pos, self.result())
        code.funcstate.release_temp(result_var)
        if self.test_if_run:
            code.putln('}')