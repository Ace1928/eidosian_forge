important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class ExprStmt(PythonBaseNode, DocstringMixin):
    type = 'expr_stmt'
    __slots__ = ()

    def get_defined_names(self, include_setitem=False):
        """
        Returns a list of `Name` defined before the `=` sign.
        """
        names = []
        if self.children[1].type == 'annassign':
            names = _defined_names(self.children[0], include_setitem)
        return [name for i in range(0, len(self.children) - 2, 2) if '=' in self.children[i + 1].value for name in _defined_names(self.children[i], include_setitem)] + names

    def get_rhs(self):
        """Returns the right-hand-side of the equals."""
        node = self.children[-1]
        if node.type == 'annassign':
            if len(node.children) == 4:
                node = node.children[3]
            else:
                node = node.children[1]
        return node

    def yield_operators(self):
        """
        Returns a generator of `+=`, `=`, etc. or None if there is no operation.
        """
        first = self.children[1]
        if first.type == 'annassign':
            if len(first.children) <= 2:
                return
            first = first.children[2]
        yield first
        yield from self.children[3::2]