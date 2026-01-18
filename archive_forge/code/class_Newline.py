important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class Newline(PythonLeaf):
    """Contains NEWLINE and ENDMARKER tokens."""
    __slots__ = ()
    type = 'newline'

    def __repr__(self):
        return '<%s: %s>' % (type(self).__name__, repr(self.value))