from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
class RecursiveNodeReplacer(VisitorTransform):
    """
    Recursively replace all occurrences of a node in a subtree by
    another node.
    """

    def __init__(self, orig_node, new_node):
        super(RecursiveNodeReplacer, self).__init__()
        self.orig_node, self.new_node = (orig_node, new_node)

    def visit_CloneNode(self, node):
        if node is self.orig_node:
            return self.new_node
        if node.arg is self.orig_node:
            node.arg = self.new_node
        return node

    def visit_Node(self, node):
        self._process_children(node)
        if node is self.orig_node:
            return self.new_node
        else:
            return node