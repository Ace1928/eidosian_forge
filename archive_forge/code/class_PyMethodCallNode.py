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
class PyMethodCallNode(SimpleCallNode):
    subexprs = ['function', 'arg_tuple']
    is_temp = True

    def generate_evaluation_code(self, code):
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        self.function.generate_evaluation_code(code)
        assert self.arg_tuple.mult_factor is None
        args = self.arg_tuple.args
        for arg in args:
            arg.generate_evaluation_code(code)
        reuse_function_temp = self.function.is_temp
        if reuse_function_temp:
            function = self.function.result()
        else:
            function = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
            self.function.make_owned_reference(code)
            code.put('%s = %s; ' % (function, self.function.py_result()))
            self.function.generate_disposal_code(code)
            self.function.free_temps(code)
        self_arg = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        code.putln('%s = NULL;' % self_arg)
        arg_offset_cname = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
        code.putln('%s = 0;' % arg_offset_cname)

        def attribute_is_likely_method(attr):
            obj = attr.obj
            if obj.is_name and obj.entry.is_pyglobal:
                return False
            return True
        if self.function.is_attribute:
            likely_method = 'likely' if attribute_is_likely_method(self.function) else 'unlikely'
        elif self.function.is_name and self.function.cf_state:
            for assignment in self.function.cf_state:
                value = assignment.rhs
                if value and value.is_attribute and value.obj.type and value.obj.type.is_pyobject:
                    if attribute_is_likely_method(value):
                        likely_method = 'likely'
                        break
            else:
                likely_method = 'unlikely'
        else:
            likely_method = 'unlikely'
        code.putln('#if CYTHON_UNPACK_METHODS')
        code.putln('if (%s(PyMethod_Check(%s))) {' % (likely_method, function))
        code.putln('%s = PyMethod_GET_SELF(%s);' % (self_arg, function))
        code.putln('if (likely(%s)) {' % self_arg)
        code.putln('PyObject* function = PyMethod_GET_FUNCTION(%s);' % function)
        code.put_incref(self_arg, py_object_type)
        code.put_incref('function', py_object_type)
        code.put_decref_set(function, py_object_type, 'function')
        code.putln('%s = 1;' % arg_offset_cname)
        code.putln('}')
        code.putln('}')
        code.putln('#endif')
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectFastCall', 'ObjectHandling.c'))
        code.putln('{')
        code.putln('PyObject *__pyx_callargs[%d] = {%s, %s};' % (len(args) + 1 if args else 2, self_arg, ', '.join((arg.py_result() for arg in args)) if args else 'NULL'))
        code.putln('%s = __Pyx_PyObject_FastCall(%s, __pyx_callargs+1-%s, %d+%s);' % (self.result(), function, arg_offset_cname, len(args), arg_offset_cname))
        code.put_xdecref_clear(self_arg, py_object_type)
        code.funcstate.release_temp(self_arg)
        code.funcstate.release_temp(arg_offset_cname)
        for arg in args:
            arg.generate_disposal_code(code)
            arg.free_temps(code)
        code.putln(code.error_goto_if_null(self.result(), self.pos))
        self.generate_gotref(code)
        if reuse_function_temp:
            self.function.generate_disposal_code(code)
            self.function.free_temps(code)
        else:
            code.put_decref_clear(function, py_object_type)
            code.funcstate.release_temp(function)
        code.putln('}')