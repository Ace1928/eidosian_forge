import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _update_positions(nodes, line_offset, last_leaf):
    for node in nodes:
        try:
            children = node.children
        except AttributeError:
            node.line += line_offset
            if node is last_leaf:
                raise _PositionUpdatingFinished
        else:
            _update_positions(children, line_offset, last_leaf)