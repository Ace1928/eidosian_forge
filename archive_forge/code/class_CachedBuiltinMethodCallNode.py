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
class CachedBuiltinMethodCallNode(CallNode):
    subexprs = ['obj', 'args']
    is_temp = True

    def __init__(self, call_node, obj, method_name, args):
        super(CachedBuiltinMethodCallNode, self).__init__(call_node.pos, obj=obj, method_name=method_name, args=args, may_return_none=call_node.may_return_none, type=call_node.type)

    def may_be_none(self):
        if self.may_return_none is not None:
            return self.may_return_none
        return ExprNode.may_be_none(self)

    def generate_result_code(self, code):
        type_cname = self.obj.type.cname
        obj_cname = self.obj.py_result()
        args = [arg.py_result() for arg in self.args]
        call_code = code.globalstate.cached_unbound_method_call_code(obj_cname, type_cname, self.method_name, args)
        code.putln('%s = %s; %s' % (self.result(), call_code, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)