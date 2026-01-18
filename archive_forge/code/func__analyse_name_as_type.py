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
def _analyse_name_as_type(name, pos, env):
    ctype = PyrexTypes.parse_basic_type(name)
    if ctype is not None and env.in_c_type_context:
        return ctype
    global_scope = env.global_scope()
    global_entry = global_scope.lookup(name)
    if global_entry and global_entry.is_type:
        type = global_entry.type
        if not env.in_c_type_context and type is Builtin.int_type and (global_scope.context.language_level == 2):
            type = py_object_type
        if type and (type.is_pyobject or env.in_c_type_context):
            return type
        ctype = ctype or type
    from .TreeFragment import TreeFragment
    with local_errors(ignore=True):
        pos = (pos[0], pos[1], pos[2] - 7)
        try:
            declaration = TreeFragment(u'sizeof(%s)' % name, name=pos[0].filename, initial_pos=pos)
        except CompileError:
            pass
        else:
            sizeof_node = declaration.root.stats[0].expr
            if isinstance(sizeof_node, SizeofTypeNode):
                sizeof_node = sizeof_node.analyse_types(env)
                if isinstance(sizeof_node, SizeofTypeNode):
                    type = sizeof_node.arg_type
                    if type and (type.is_pyobject or env.in_c_type_context):
                        return type
                    ctype = ctype or type
    return ctype