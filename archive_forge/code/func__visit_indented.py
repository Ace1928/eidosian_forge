from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def _visit_indented(self, node):
    self.indent()
    self.visit(node)
    self.dedent()