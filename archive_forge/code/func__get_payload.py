important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def _get_payload(self):
    match = re.search('(\'{3}|"{3}|\'|")(.*)$', self.value, flags=re.DOTALL)
    return match.group(2)[:-len(match.group(1))]