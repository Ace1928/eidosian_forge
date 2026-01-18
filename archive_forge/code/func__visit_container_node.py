from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def _visit_container_node(self, node, decl, extras, attributes):
    self.startline(decl)
    if node.name:
        self.put(u' ')
        self.put(node.name)
        if node.cname is not None:
            self.put(u' "%s"' % node.cname)
    if extras:
        self.put(extras)
    self.endline(':')
    self.indent()
    if not attributes:
        self.putline('pass')
    else:
        for attribute in attributes:
            self.visit(attribute)
    self.dedent()