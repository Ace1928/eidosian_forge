from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class SetIterationNextNode(Node):
    child_attrs = ['set_obj', 'expected_size', 'pos_index_var', 'coerced_value_var', 'value_target', 'is_set_flag']
    coerced_value_var = value_ref = None

    def __init__(self, set_obj, expected_size, pos_index_var, value_target, is_set_flag):
        Node.__init__(self, set_obj.pos, set_obj=set_obj, expected_size=expected_size, pos_index_var=pos_index_var, value_target=value_target, is_set_flag=is_set_flag, is_temp=True, type=PyrexTypes.c_bint_type)

    def analyse_expressions(self, env):
        from . import ExprNodes
        self.set_obj = self.set_obj.analyse_types(env)
        self.expected_size = self.expected_size.analyse_types(env)
        self.pos_index_var = self.pos_index_var.analyse_types(env)
        self.value_target = self.value_target.analyse_target_types(env)
        self.value_ref = ExprNodes.TempNode(self.value_target.pos, type=PyrexTypes.py_object_type)
        self.coerced_value_var = self.value_ref.coerce_to(self.value_target.type, env)
        self.is_set_flag = self.is_set_flag.analyse_types(env)
        return self

    def generate_function_definitions(self, env, code):
        self.set_obj.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        code.globalstate.use_utility_code(UtilityCode.load_cached('set_iter', 'Optimize.c'))
        self.set_obj.generate_evaluation_code(code)
        value_ref = self.value_ref
        value_ref.allocate(code)
        result_temp = code.funcstate.allocate_temp(PyrexTypes.c_int_type, False)
        code.putln('%s = __Pyx_set_iter_next(%s, %s, &%s, &%s, %s);' % (result_temp, self.set_obj.py_result(), self.expected_size.result(), self.pos_index_var.result(), value_ref.result(), self.is_set_flag.result()))
        code.putln('if (unlikely(%s == 0)) break;' % result_temp)
        code.putln(code.error_goto_if('%s == -1' % result_temp, self.pos))
        code.funcstate.release_temp(result_temp)
        value_ref.generate_gotref(code)
        self.coerced_value_var.generate_evaluation_code(code)
        self.value_target.generate_assignment_code(self.coerced_value_var, code)
        value_ref.release(code)