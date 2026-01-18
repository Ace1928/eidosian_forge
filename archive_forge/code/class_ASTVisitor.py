from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
class ASTVisitor(object):
    """Performs a depth-first walk of the AST."""

    def __init__(self):
        self.node = None
        self._cache = {}

    def default(self, node, *args):
        for child in iter_child_nodes(node):
            self.dispatch(child, *args)

    def dispatch(self, node, *args):
        self.node = node
        klass = node.__class__
        meth = self._cache.get(klass)
        if meth is None:
            className = klass.__name__
            meth = getattr(self.visitor, 'visit' + className, self.default)
            self._cache[klass] = meth
        return meth(node, *args)

    def preorder(self, tree, visitor, *args):
        """Do preorder walk of tree using visitor"""
        self.visitor = visitor
        visitor.visit = self.dispatch
        self.dispatch(tree, *args)