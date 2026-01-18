important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class EndMarker(_LeafWithoutNewlines):
    __slots__ = ()
    type = 'endmarker'

    def __repr__(self):
        return '<%s: prefix=%s end_pos=%s>' % (type(self).__name__, repr(self.prefix), self.end_pos)