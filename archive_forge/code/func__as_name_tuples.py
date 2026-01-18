important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def _as_name_tuples(self):
    last = self.children[-1]
    if last == ')':
        last = self.children[-2]
    elif last == '*':
        return
    if last.type == 'import_as_names':
        as_names = last.children[::2]
    else:
        as_names = [last]
    for as_name in as_names:
        if as_name.type == 'name':
            yield (as_name, None)
        else:
            yield as_name.children[::2]