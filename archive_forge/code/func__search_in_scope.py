important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def _search_in_scope(self, *names):

    def scan(children):
        for element in children:
            if element.type in names:
                yield element
            if element.type in _FUNC_CONTAINERS:
                yield from scan(element.children)
    return scan(self.children)