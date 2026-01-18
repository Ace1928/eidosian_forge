import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _get_matching_indent_nodes(self, tree_nodes, is_new_suite):
    node_iterator = iter(tree_nodes)
    if is_new_suite:
        yield next(node_iterator)
    first_node = next(node_iterator)
    indent = _get_indentation(first_node)
    if not is_new_suite and indent not in self.indents:
        return
    yield first_node
    for n in node_iterator:
        if _get_indentation(n) != indent:
            return
        yield n