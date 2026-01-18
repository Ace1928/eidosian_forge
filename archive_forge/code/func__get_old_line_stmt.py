import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _get_old_line_stmt(self, old_line):
    leaf = self._module.get_leaf_for_position((old_line, 0), include_prefixes=True)
    if _ends_with_newline(leaf):
        leaf = leaf.get_next_leaf()
    if leaf.get_start_pos_of_prefix()[0] == old_line:
        node = leaf
        while node.parent.type not in ('file_input', 'suite'):
            node = node.parent
        if node.start_pos[0] >= old_line:
            return node
    return None