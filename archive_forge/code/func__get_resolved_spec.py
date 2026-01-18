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
def _get_resolved_spec(env, spec):
    if isinstance(spec, NameNode):
        return _resolve_NameNode(env, spec)
    elif isinstance(spec, AttributeNode):
        return _resolve_AttributeNode(env, spec)
    else:
        raise CompileError(spec.pos, INVALID_ERR)