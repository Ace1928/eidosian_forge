important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class NamedExpr(PythonBaseNode):
    type = 'namedexpr_test'

    def get_defined_names(self, include_setitem=False):
        return _defined_names(self.children[0], include_setitem)