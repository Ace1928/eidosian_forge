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
class BoolBinopResultNode(ExprNode):
    """
    Intermediate result of a short-circuiting and/or expression.
    Tests the result for 'truthiness' and takes care of coercing the final result
    of the overall expression to the target type.

    Note that this node provides the same code generation method as
    BoolBinopNode to simplify expression nesting.

    arg     ExprNode    the argument to test
    value   ExprNode    the coerced result value node
    """
    subexprs = ['arg', 'value']
    is_temp = True
    arg = None
    value = None

    def __init__(self, arg, result_type, env):
        arg = arg.coerce_to_simple(env)
        arg = ProxyNode(arg)
        super(BoolBinopResultNode, self).__init__(arg.pos, arg=arg, type=result_type, value=CloneNode(arg).coerce_to(result_type, env))

    def coerce_to_boolean(self, env):
        return self.coerce_to(PyrexTypes.c_bint_type, env)

    def coerce_to(self, dst_type, env):
        arg = self.arg.arg
        if dst_type is PyrexTypes.c_bint_type:
            arg = arg.coerce_to_boolean(env)
        return BoolBinopResultNode(arg, dst_type, env)

    def nogil_check(self, env):
        pass

    def generate_operand_test(self, code):
        if self.arg.type.is_pyobject:
            test_result = code.funcstate.allocate_temp(PyrexTypes.c_bint_type, manage_ref=False)
            code.putln('%s = __Pyx_PyObject_IsTrue(%s); %s' % (test_result, self.arg.py_result(), code.error_goto_if_neg(test_result, self.pos)))
        else:
            test_result = self.arg.result()
        return (test_result, self.arg.type.is_pyobject)

    def generate_bool_evaluation_code(self, code, final_result_temp, final_result_type, and_label, or_label, end_label, fall_through):
        code.mark_pos(self.pos)
        self.arg.generate_evaluation_code(code)
        if and_label or or_label:
            test_result, uses_temp = self.generate_operand_test(code)
            if uses_temp and (and_label and or_label):
                self.arg.generate_disposal_code(code)
            sense = '!' if or_label else ''
            code.putln('if (%s%s) {' % (sense, test_result))
            if uses_temp:
                code.funcstate.release_temp(test_result)
            if not uses_temp or not (and_label and or_label):
                self.arg.generate_disposal_code(code)
            if or_label and or_label != fall_through:
                code.put_goto(or_label)
            if and_label:
                if or_label:
                    code.putln('} else {')
                    if not uses_temp:
                        self.arg.generate_disposal_code(code)
                if and_label != fall_through:
                    code.put_goto(and_label)
        if not and_label or not or_label:
            if and_label or or_label:
                code.putln('} else {')
            self.value.generate_evaluation_code(code)
            self.value.make_owned_reference(code)
            code.putln('%s = %s;' % (final_result_temp, self.value.result_as(final_result_type)))
            self.value.generate_post_assignment_code(code)
            self.arg.generate_disposal_code(code)
            self.value.free_temps(code)
            if end_label != fall_through:
                code.put_goto(end_label)
        if and_label or or_label:
            code.putln('}')
        self.arg.free_temps(code)

    def analyse_types(self, env):
        return self