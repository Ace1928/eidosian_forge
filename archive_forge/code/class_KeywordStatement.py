important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class KeywordStatement(PythonBaseNode):
    """
    For the following statements: `assert`, `del`, `global`, `nonlocal`,
    `raise`, `return`, `yield`.

    `pass`, `continue` and `break` are not in there, because they are just
    simple keywords and the parser reduces it to a keyword.
    """
    __slots__ = ()

    @property
    def type(self):
        """
        Keyword statements start with the keyword and end with `_stmt`. You can
        crosscheck this with the Python grammar.
        """
        return '%s_stmt' % self.keyword

    @property
    def keyword(self):
        return self.children[0].value

    def get_defined_names(self, include_setitem=False):
        keyword = self.keyword
        if keyword == 'del':
            return _defined_names(self.children[1], include_setitem)
        if keyword in ('global', 'nonlocal'):
            return self.children[1::2]
        return []