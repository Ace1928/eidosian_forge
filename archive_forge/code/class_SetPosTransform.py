from __future__ import absolute_import
import re
from io import StringIO
from .Scanning import PyrexScanner, StringSourceDescriptor
from .Symtab import ModuleScope
from . import PyrexTypes
from .Visitor import VisitorTransform
from .Nodes import Node, StatListNode
from .ExprNodes import NameNode
from .StringEncoding import _unicode
from . import Parsing
from . import Main
from . import UtilNodes
class SetPosTransform(VisitorTransform):

    def __init__(self, pos):
        super(SetPosTransform, self).__init__()
        self.pos = pos

    def visit_Node(self, node):
        node.pos = self.pos
        self.visitchildren(node)
        return node