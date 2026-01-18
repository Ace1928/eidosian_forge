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
class JoinedStrNode(ExprNode):
    type = unicode_type
    is_temp = True
    gil_message = 'String concatenation'
    subexprs = ['values']

    def analyse_types(self, env):
        self.values = [v.analyse_types(env).coerce_to_pyobject(env) for v in self.values]
        return self

    def may_be_none(self):
        return False

    def generate_evaluation_code(self, code):
        code.mark_pos(self.pos)
        num_items = len(self.values)
        list_var = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        ulength_var = code.funcstate.allocate_temp(PyrexTypes.c_py_ssize_t_type, manage_ref=False)
        max_char_var = code.funcstate.allocate_temp(PyrexTypes.c_py_ucs4_type, manage_ref=False)
        code.putln('%s = PyTuple_New(%s); %s' % (list_var, num_items, code.error_goto_if_null(list_var, self.pos)))
        code.put_gotref(list_var, py_object_type)
        code.putln('%s = 0;' % ulength_var)
        code.putln('%s = 127;' % max_char_var)
        for i, node in enumerate(self.values):
            node.generate_evaluation_code(code)
            node.make_owned_reference(code)
            ulength = '__Pyx_PyUnicode_GET_LENGTH(%s)' % node.py_result()
            max_char_value = '__Pyx_PyUnicode_MAX_CHAR_VALUE(%s)' % node.py_result()
            is_ascii = False
            if isinstance(node, UnicodeNode):
                try:
                    node.value.encode('iso8859-1')
                    max_char_value = '255'
                    node.value.encode('us-ascii')
                    is_ascii = True
                except UnicodeEncodeError:
                    if max_char_value != '255':
                        max_char = max(map(ord, node.value))
                        if max_char < 55296:
                            max_char_value = '65535'
                            ulength = str(len(node.value))
                        elif max_char >= 65536:
                            max_char_value = '1114111'
                            ulength = str(len(node.value))
                        else:
                            pass
                else:
                    ulength = str(len(node.value))
            elif isinstance(node, FormattedValueNode) and node.value.type.is_numeric:
                is_ascii = True
            if not is_ascii:
                code.putln('%s = (%s > %s) ? %s : %s;' % (max_char_var, max_char_value, max_char_var, max_char_value, max_char_var))
            code.putln('%s += %s;' % (ulength_var, ulength))
            node.generate_giveref(code)
            code.putln('PyTuple_SET_ITEM(%s, %s, %s);' % (list_var, i, node.py_result()))
            node.generate_post_assignment_code(code)
            node.free_temps(code)
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        code.globalstate.use_utility_code(UtilityCode.load_cached('JoinPyUnicode', 'StringTools.c'))
        code.putln('%s = __Pyx_PyUnicode_Join(%s, %d, %s, %s); %s' % (self.result(), list_var, num_items, ulength_var, max_char_var, code.error_goto_if_null(self.py_result(), self.pos)))
        self.generate_gotref(code)
        code.put_decref_clear(list_var, py_object_type)
        code.funcstate.release_temp(list_var)
        code.funcstate.release_temp(ulength_var)
        code.funcstate.release_temp(max_char_var)