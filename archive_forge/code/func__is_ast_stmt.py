import sys
import ast
import py
from py._code.assertion import _format_explanation, BuiltinAssertionError
def _is_ast_stmt(node):
    return isinstance(node, ast.stmt)