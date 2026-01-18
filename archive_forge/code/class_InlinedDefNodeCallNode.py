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
class InlinedDefNodeCallNode(CallNode):
    subexprs = ['args', 'function_name']
    is_temp = 1
    type = py_object_type
    function = None
    function_name = None

    def can_be_inlined(self):
        func_type = self.function.def_node
        if func_type.star_arg or func_type.starstar_arg:
            return False
        if len(func_type.args) != len(self.args):
            return False
        if func_type.num_kwonly_args:
            return False
        return True

    def analyse_types(self, env):
        self.function_name = self.function_name.analyse_types(env)
        self.args = [arg.analyse_types(env) for arg in self.args]
        func_type = self.function.def_node
        actual_nargs = len(self.args)
        some_args_in_temps = False
        for i in range(actual_nargs):
            formal_type = func_type.args[i].type
            arg = self.args[i].coerce_to(formal_type, env)
            if arg.is_temp:
                if i > 0:
                    some_args_in_temps = True
            elif arg.type.is_pyobject and (not env.nogil):
                if arg.nonlocally_immutable():
                    pass
                else:
                    if i > 0:
                        some_args_in_temps = True
                    arg = arg.coerce_to_temp(env)
            self.args[i] = arg
        if some_args_in_temps:
            for i in range(actual_nargs - 1):
                arg = self.args[i]
                if arg.nonlocally_immutable():
                    pass
                elif arg.type.is_cpp_class:
                    pass
                elif env.nogil and arg.type.is_pyobject:
                    pass
                elif i > 0:
                    warning(arg.pos, 'Argument evaluation order in C function call is undefined and may not be as expected', 0)
                    break
        return self

    def generate_result_code(self, code):
        arg_code = [self.function_name.py_result()]
        func_type = self.function.def_node
        for arg, proto_arg in zip(self.args, func_type.args):
            if arg.type.is_pyobject:
                arg_code.append(arg.result_as(proto_arg.type))
            else:
                arg_code.append(arg.result())
        arg_code = ', '.join(arg_code)
        code.putln('%s = %s(%s); %s' % (self.result(), self.function.def_node.entry.pyfunc_cname, arg_code, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)