from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def _resolve_NameNode(env, node):
    try:
        resolved_name = env.lookup(node.name).name
    except AttributeError:
        raise CompileError(node.pos, INVALID_ERR)
    viewscope = env.global_scope().context.cython_scope.viewscope
    entry = viewscope.lookup(resolved_name)
    if entry is None:
        raise CompileError(node.pos, NOT_CIMPORTED_ERR)
    return entry