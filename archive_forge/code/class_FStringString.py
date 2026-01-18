important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class FStringString(PythonLeaf):
    """
    f-strings contain f-string expressions and normal python strings. These are
    the string parts of f-strings.
    """
    type = 'fstring_string'
    __slots__ = ()