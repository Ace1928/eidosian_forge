from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def _handle_fused(self, node):
    if node.is_generator and node.has_fused_arguments:
        node.has_fused_arguments = False
        error(node.pos, 'Fused generators not supported')
        node.gbody = Nodes.StatListNode(node.pos, stats=[], body=Nodes.PassStatNode(node.pos))
    return node.has_fused_arguments