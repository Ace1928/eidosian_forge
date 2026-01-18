import ast
import inspect
import io
import linecache
import re
import sys
import textwrap
import tokenize
import astunparse
import gast
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.util import tf_inspect
def _without_context(node, lines, minl, maxl):
    """Returns a clean node and source code without indenting and context."""
    for n in gast.walk(node):
        lineno = getattr(n, 'lineno', None)
        if lineno is not None:
            n.lineno = lineno - minl
        end_lineno = getattr(n, 'end_lineno', None)
        if end_lineno is not None:
            n.end_lineno = end_lineno - minl
    code_lines = lines[minl - 1:maxl]
    end_col_offset = getattr(node, 'end_col_offset', None)
    if end_col_offset is not None:
        code_lines[-1] = code_lines[-1][:end_col_offset]
    col_offset = getattr(node, 'col_offset', None)
    if col_offset is None:
        match = re.search('(?<!\\w)lambda(?!\\w)', code_lines[0])
        if match is not None:
            col_offset = match.start(0)
    if col_offset is not None:
        code_lines[0] = code_lines[0][col_offset:]
    code_block = '\n'.join([c.rstrip() for c in code_lines])
    return (node, code_block)