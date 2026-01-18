import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _remove_endmarker(self, tree_nodes):
    """
        Helps cleaning up the tree nodes that get inserted.
        """
    last_leaf = tree_nodes[-1].get_last_leaf()
    is_endmarker = last_leaf.type == 'endmarker'
    self._prefix_remainder = ''
    if is_endmarker:
        prefix = last_leaf.prefix
        separation = max(prefix.rfind('\n'), prefix.rfind('\r'))
        if separation > -1:
            last_leaf.prefix, self._prefix_remainder = (last_leaf.prefix[:separation + 1], last_leaf.prefix[separation + 1:])
    self.prefix = ''
    if is_endmarker:
        self.prefix = last_leaf.prefix
        tree_nodes = tree_nodes[:-1]
    return tree_nodes