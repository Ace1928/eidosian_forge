import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _func_or_class_has_suite(node):
    if node.type == 'decorated':
        node = node.children[-1]
    if node.type in ('async_funcdef', 'async_stmt'):
        node = node.children[-1]
    return node.type in ('classdef', 'funcdef') and node.children[-1].type == 'suite'