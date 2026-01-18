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
class TreeCopier(VisitorTransform):

    def visit_Node(self, node):
        if node is None:
            return node
        else:
            c = node.clone_node()
            self.visitchildren(c)
            return c