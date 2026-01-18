important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class ClassOrFunc(Scope):
    __slots__ = ()

    @property
    def name(self):
        """
        Returns the `Name` leaf that defines the function or class name.
        """
        return self.children[1]

    def get_decorators(self):
        """
        :rtype: list of :class:`Decorator`
        """
        decorated = self.parent
        if decorated.type == 'async_funcdef':
            decorated = decorated.parent
        if decorated.type == 'decorated':
            if decorated.children[0].type == 'decorators':
                return decorated.children[0].children
            else:
                return decorated.children[:1]
        else:
            return []