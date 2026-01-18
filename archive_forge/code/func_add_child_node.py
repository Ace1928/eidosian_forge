import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def add_child_node(self, child_node):
    self._node_children.append(child_node)