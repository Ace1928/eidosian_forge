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
class YieldNodeCollector(TreeVisitor):

    def __init__(self, excludes=[]):
        super(YieldNodeCollector, self).__init__()
        self.yields = []
        self.returns = []
        self.finallys = []
        self.excepts = []
        self.has_return_value = False
        self.has_yield = False
        self.has_await = False
        self.excludes = excludes

    def visit_Node(self, node):
        if node not in self.excludes:
            self.visitchildren(node)

    def visit_YieldExprNode(self, node):
        self.yields.append(node)
        self.has_yield = True
        self.visitchildren(node)

    def visit_AwaitExprNode(self, node):
        self.yields.append(node)
        self.has_await = True
        self.visitchildren(node)

    def visit_ReturnStatNode(self, node):
        self.visitchildren(node)
        if node.value:
            self.has_return_value = True
        self.returns.append(node)

    def visit_TryFinallyStatNode(self, node):
        self.visitchildren(node)
        self.finallys.append(node)

    def visit_TryExceptStatNode(self, node):
        self.visitchildren(node)
        self.excepts.append(node)

    def visit_ClassDefNode(self, node):
        pass

    def visit_FuncDefNode(self, node):
        pass

    def visit_LambdaNode(self, node):
        pass

    def visit_GeneratorExpressionNode(self, node):
        if isinstance(node.loop, Nodes._ForInStatNode):
            self.visit(node.loop.iterator)

    def visit_CArgDeclNode(self, node):
        pass