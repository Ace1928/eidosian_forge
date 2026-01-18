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
class MergedSequenceNode(ExprNode):
    """
    Merge a sequence of iterables into a set/list/tuple.

    The target collection is determined by self.type, which must be set externally.

    args    [ExprNode]
    """
    subexprs = ['args']
    is_temp = True
    gil_message = 'Constructing Python collection'

    def __init__(self, pos, args, type):
        if type in (list_type, tuple_type) and args and args[0].is_sequence_constructor:
            if args[0].type is not list_type:
                args[0] = ListNode(args[0].pos, args=args[0].args, is_temp=True, mult_factor=args[0].mult_factor)
        ExprNode.__init__(self, pos, args=args, type=type)

    def calculate_constant_result(self):
        result = []
        for item in self.args:
            if item.is_sequence_constructor and item.mult_factor:
                if item.mult_factor.constant_result <= 0:
                    continue
            if item.is_set_literal or item.is_sequence_constructor:
                items = (arg.constant_result for arg in item.args)
            else:
                items = item.constant_result
            result.extend(items)
        if self.type is set_type:
            result = set(result)
        elif self.type is tuple_type:
            result = tuple(result)
        else:
            assert self.type is list_type
        self.constant_result = result

    def compile_time_value(self, denv):
        result = []
        for item in self.args:
            if item.is_sequence_constructor and item.mult_factor:
                if item.mult_factor.compile_time_value(denv) <= 0:
                    continue
            if item.is_set_literal or item.is_sequence_constructor:
                items = (arg.compile_time_value(denv) for arg in item.args)
            else:
                items = item.compile_time_value(denv)
            result.extend(items)
        if self.type is set_type:
            try:
                result = set(result)
            except Exception as e:
                self.compile_time_value_error(e)
        elif self.type is tuple_type:
            result = tuple(result)
        else:
            assert self.type is list_type
        return result

    def type_dependencies(self, env):
        return ()

    def infer_type(self, env):
        return self.type

    def analyse_types(self, env):
        args = [arg.analyse_types(env).coerce_to_pyobject(env).as_none_safe_node('argument after * must be an iterable, not NoneType') for arg in self.args]
        if len(args) == 1 and args[0].type is self.type:
            return args[0]
        assert self.type in (set_type, list_type, tuple_type)
        self.args = args
        return self

    def may_be_none(self):
        return False

    def generate_evaluation_code(self, code):
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        is_set = self.type is set_type
        args = iter(self.args)
        item = next(args)
        item.generate_evaluation_code(code)
        if is_set and item.is_set_literal or (not is_set and item.is_sequence_constructor and (item.type is list_type)):
            code.putln('%s = %s;' % (self.result(), item.py_result()))
            item.generate_post_assignment_code(code)
        else:
            code.putln('%s = %s(%s); %s' % (self.result(), 'PySet_New' if is_set else '__Pyx_PySequence_ListKeepNew' if item.is_temp and item.type in (py_object_type, list_type) else 'PySequence_List', item.py_result(), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            item.generate_disposal_code(code)
        item.free_temps(code)
        helpers = set()
        if is_set:
            add_func = 'PySet_Add'
            extend_func = '__Pyx_PySet_Update'
        else:
            add_func = '__Pyx_ListComp_Append'
            extend_func = '__Pyx_PyList_Extend'
        for item in args:
            if is_set and (item.is_set_literal or item.is_sequence_constructor) or (item.is_sequence_constructor and (not item.mult_factor)):
                if not is_set and item.args:
                    helpers.add(('ListCompAppend', 'Optimize.c'))
                for arg in item.args:
                    arg.generate_evaluation_code(code)
                    code.put_error_if_neg(arg.pos, '%s(%s, %s)' % (add_func, self.result(), arg.py_result()))
                    arg.generate_disposal_code(code)
                    arg.free_temps(code)
                continue
            if is_set:
                helpers.add(('PySet_Update', 'Builtins.c'))
            else:
                helpers.add(('ListExtend', 'Optimize.c'))
            item.generate_evaluation_code(code)
            code.put_error_if_neg(item.pos, '%s(%s, %s)' % (extend_func, self.result(), item.py_result()))
            item.generate_disposal_code(code)
            item.free_temps(code)
        if self.type is tuple_type:
            code.putln('{')
            code.putln('PyObject *%s = PyList_AsTuple(%s);' % (Naming.quick_temp_cname, self.result()))
            code.put_decref(self.result(), py_object_type)
            code.putln('%s = %s; %s' % (self.result(), Naming.quick_temp_cname, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            code.putln('}')
        for helper in sorted(helpers):
            code.globalstate.use_utility_code(UtilityCode.load_cached(*helper))

    def annotate(self, code):
        for item in self.args:
            item.annotate(code)