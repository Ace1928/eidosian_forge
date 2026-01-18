import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _flows_finished(pgen_grammar, stack):
    """
    if, while, for and try might not be finished, because another part might
    still be parsed.
    """
    for stack_node in stack:
        if stack_node.nonterminal in ('if_stmt', 'while_stmt', 'for_stmt', 'try_stmt'):
            return False
    return True