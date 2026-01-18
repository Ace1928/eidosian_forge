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
class CodeObjectNode(ExprNode):
    subexprs = ['varnames']
    is_temp = False
    result_code = None

    def __init__(self, def_node):
        ExprNode.__init__(self, def_node.pos, def_node=def_node)
        args = list(def_node.args)
        local_vars = [arg for arg in def_node.local_scope.var_entries if arg.name]
        self.varnames = TupleNode(def_node.pos, args=[IdentifierStringNode(arg.pos, value=arg.name) for arg in args + local_vars], is_temp=0, is_literal=1)

    def may_be_none(self):
        return False

    def calculate_result_code(self, code=None):
        if self.result_code is None:
            self.result_code = code.get_py_const(py_object_type, 'codeobj', cleanup_level=2)
        return self.result_code

    def generate_result_code(self, code):
        if self.result_code is None:
            self.result_code = code.get_py_const(py_object_type, 'codeobj', cleanup_level=2)
        code = code.get_cached_constants_writer(self.result_code)
        if code is None:
            return
        code.mark_pos(self.pos)
        func = self.def_node
        func_name = code.get_py_string_const(func.name, identifier=True, is_str=False, unicode_value=func.name)
        file_path = StringEncoding.bytes_literal(func.pos[0].get_filenametable_entry().encode('utf8'), 'utf8')
        file_path_const = code.get_py_string_const(file_path, identifier=False, is_str=True)
        flags = ['CO_OPTIMIZED', 'CO_NEWLOCALS']
        if self.def_node.star_arg:
            flags.append('CO_VARARGS')
        if self.def_node.starstar_arg:
            flags.append('CO_VARKEYWORDS')
        if self.def_node.is_asyncgen:
            flags.append('CO_ASYNC_GENERATOR')
        elif self.def_node.is_coroutine:
            flags.append('CO_COROUTINE')
        elif self.def_node.is_generator:
            flags.append('CO_GENERATOR')
        code.putln('%s = (PyObject*)__Pyx_PyCode_New(%d, %d, %d, %d, 0, %s, %s, %s, %s, %s, %s, %s, %s, %s, %d, %s); %s' % (self.result_code, len(func.args) - func.num_kwonly_args, func.num_posonly_args, func.num_kwonly_args, len(self.varnames.args), '|'.join(flags) or '0', Naming.empty_bytes, Naming.empty_tuple, Naming.empty_tuple, self.varnames.result(), Naming.empty_tuple, Naming.empty_tuple, file_path_const, func_name, self.pos[1], Naming.empty_bytes, code.error_goto_if_null(self.result_code, self.pos)))