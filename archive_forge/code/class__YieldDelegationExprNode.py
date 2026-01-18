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
class _YieldDelegationExprNode(YieldExprNode):

    def yield_from_func(self, code):
        raise NotImplementedError()

    def generate_evaluation_code(self, code, source_cname=None, decref_source=False):
        if source_cname is None:
            self.arg.generate_evaluation_code(code)
        code.putln('%s = %s(%s, %s);' % (Naming.retval_cname, self.yield_from_func(code), Naming.generator_cname, self.arg.py_result() if source_cname is None else source_cname))
        if source_cname is None:
            self.arg.generate_disposal_code(code)
            self.arg.free_temps(code)
        elif decref_source:
            code.put_decref_clear(source_cname, py_object_type)
        code.put_xgotref(Naming.retval_cname, py_object_type)
        code.putln('if (likely(%s)) {' % Naming.retval_cname)
        self.generate_yield_code(code)
        code.putln('} else {')
        if self.result_is_used:
            self.fetch_iteration_result(code)
        else:
            self.handle_iteration_exception(code)
        code.putln('}')

    def fetch_iteration_result(self, code):
        code.putln('%s = NULL;' % self.result())
        code.put_error_if_neg(self.pos, '__Pyx_PyGen_FetchStopIterationValue(&%s)' % self.result())
        self.generate_gotref(code)

    def handle_iteration_exception(self, code):
        code.putln('PyObject* exc_type = __Pyx_PyErr_CurrentExceptionType();')
        code.putln('if (exc_type) {')
        code.putln('if (likely(exc_type == PyExc_StopIteration || (exc_type != PyExc_GeneratorExit && __Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration)))) PyErr_Clear();')
        code.putln('else %s' % code.error_goto(self.pos))
        code.putln('}')