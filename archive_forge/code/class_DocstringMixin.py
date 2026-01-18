important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class DocstringMixin:
    __slots__ = ()

    def get_doc_node(self):
        """
        Returns the string leaf of a docstring. e.g. ``r'''foo'''``.
        """
        if self.type == 'file_input':
            node = self.children[0]
        elif self.type in ('funcdef', 'classdef'):
            node = self.children[self.children.index(':') + 1]
            if node.type == 'suite':
                node = node.children[1]
        else:
            simple_stmt = self.parent
            c = simple_stmt.parent.children
            index = c.index(simple_stmt)
            if not index:
                return None
            node = c[index - 1]
        if node.type == 'simple_stmt':
            node = node.children[0]
        if node.type == 'string':
            return node
        return None