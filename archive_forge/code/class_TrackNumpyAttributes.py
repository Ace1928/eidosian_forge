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
class TrackNumpyAttributes(VisitorTransform, SkipDeclarations):

    def __init__(self):
        super(TrackNumpyAttributes, self).__init__()
        self.numpy_module_names = set()

    def visit_CImportStatNode(self, node):
        if node.module_name == u'numpy':
            self.numpy_module_names.add(node.as_name or u'numpy')
        return node

    def visit_AttributeNode(self, node):
        self.visitchildren(node)
        obj = node.obj
        if obj.is_name and obj.name in self.numpy_module_names or obj.is_numpy_attribute:
            node.is_numpy_attribute = True
        return node
    visit_Node = VisitorTransform.recurse_to_children