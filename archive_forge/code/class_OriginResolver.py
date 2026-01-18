import collections
import difflib
import io
import os
import tokenize
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.util import tf_inspect
class OriginResolver(gast.NodeVisitor):
    """Annotates an AST with additional source information like file name."""

    def __init__(self, root_node, source_lines, comments_map, context_lineno, context_col_offset, filepath):
        self._source_lines = source_lines
        self._comments_map = comments_map
        if hasattr(root_node, 'decorator_list') and root_node.decorator_list and hasattr(root_node.decorator_list[0], 'lineno'):
            self._lineno_offset = context_lineno - root_node.decorator_list[0].lineno
        else:
            self._lineno_offset = context_lineno - root_node.lineno
        self._col_offset = context_col_offset - root_node.col_offset
        self._filepath = filepath
        self._function_stack = []

    def _absolute_lineno(self, lineno):
        return lineno + self._lineno_offset

    def _absolute_col_offset(self, col_offset):
        if col_offset is None:
            return 0
        return col_offset + self._col_offset

    def _attach_origin_info(self, node):
        lineno = getattr(node, 'lineno', None)
        col_offset = getattr(node, 'col_offset', None)
        if lineno is None:
            return
        if self._function_stack:
            function_name = self._function_stack[-1].name
        else:
            function_name = None
        source_code_line = self._source_lines[lineno - 1]
        comment = self._comments_map.get(lineno)
        loc = Location(self._filepath, self._absolute_lineno(lineno), self._absolute_col_offset(col_offset))
        origin = OriginInfo(loc, function_name, source_code_line, comment)
        anno.setanno(node, 'lineno', lineno)
        anno.setanno(node, anno.Basic.ORIGIN, origin)

    def visit(self, node):
        entered_function = False
        if isinstance(node, gast.FunctionDef):
            entered_function = True
            self._function_stack.append(_Function(node.name))
        self._attach_origin_info(node)
        self.generic_visit(node)
        if entered_function:
            self._function_stack.pop()