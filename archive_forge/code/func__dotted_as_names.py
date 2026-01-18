important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def _dotted_as_names(self):
    """Generator of (list(path), alias) where alias may be None."""
    dotted_as_names = self.children[1]
    if dotted_as_names.type == 'dotted_as_names':
        as_names = dotted_as_names.children[::2]
    else:
        as_names = [dotted_as_names]
    for as_name in as_names:
        if as_name.type == 'dotted_as_name':
            alias = as_name.children[2]
            as_name = as_name.children[0]
        else:
            alias = None
        if as_name.type == 'name':
            yield ([as_name], alias)
        else:
            yield (as_name.children[::2], alias)