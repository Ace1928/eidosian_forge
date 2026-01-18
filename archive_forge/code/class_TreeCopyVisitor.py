import sys
import os
import re
import warnings
import types
import unicodedata
class TreeCopyVisitor(GenericNodeVisitor):
    """
    Make a complete copy of a tree or branch, including element attributes.
    """

    def __init__(self, document):
        GenericNodeVisitor.__init__(self, document)
        self.parent_stack = []
        self.parent = []

    def get_tree_copy(self):
        return self.parent[0]

    def default_visit(self, node):
        """Copy the current node, and make it the new acting parent."""
        newnode = node.copy()
        self.parent.append(newnode)
        self.parent_stack.append(self.parent)
        self.parent = newnode

    def default_departure(self, node):
        """Restore the previous acting parent."""
        self.parent = self.parent_stack.pop()