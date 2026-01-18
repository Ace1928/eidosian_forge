import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _is_flow_node(node):
    if node.type == 'async_stmt':
        node = node.children[1]
    try:
        value = node.children[0].value
    except AttributeError:
        return False
    return value in ('if', 'for', 'while', 'try', 'with')