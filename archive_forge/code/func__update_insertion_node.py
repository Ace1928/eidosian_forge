import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _update_insertion_node(self, indentation):
    for node in reversed(list(self._working_stack)):
        if node.indentation < indentation or node is self._working_stack[0]:
            return node
        self._working_stack.pop()