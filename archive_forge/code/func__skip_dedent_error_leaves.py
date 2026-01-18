import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _skip_dedent_error_leaves(leaf):
    while leaf is not None and leaf.type == 'error_leaf' and (leaf.token_type == 'DEDENT'):
        leaf = leaf.get_previous_leaf()
    return leaf