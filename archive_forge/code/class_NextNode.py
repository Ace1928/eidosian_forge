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
class NextNode(AtomicExprNode):

    def __init__(self, iterator):
        AtomicExprNode.__init__(self, iterator.pos)
        self.iterator = iterator

    def nogil_check(self, env):
        pass

    def type_dependencies(self, env):
        return self.iterator.type_dependencies(env)

    def infer_type(self, env, iterator_type=None):
        if iterator_type is None:
            iterator_type = self.iterator.infer_type(env)
        if iterator_type.is_ptr or iterator_type.is_array:
            return iterator_type.base_type
        elif iterator_type.is_cpp_class:
            item_type = env.lookup_operator_for_types(self.pos, '*', [iterator_type]).type.return_type
            item_type = PyrexTypes.remove_cv_ref(item_type, remove_fakeref=True)
            return item_type
        else:
            fake_index_node = IndexNode(self.pos, base=self.iterator.sequence, index=IntNode(self.pos, value='PY_SSIZE_T_MAX', type=PyrexTypes.c_py_ssize_t_type))
            return fake_index_node.infer_type(env)

    def analyse_types(self, env):
        self.type = self.infer_type(env, self.iterator.type)
        self.is_temp = 1
        return self

    def generate_result_code(self, code):
        self.iterator.generate_iter_next_result_code(self.result(), code)