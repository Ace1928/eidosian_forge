from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED, EXCEPTION_TYPE_UNHANDLED, \
from _pydev_bundle import pydev_log
import itertools
from typing import Any, Dict
def _extract_caret_anchors_in_bytes_from_line_segment(segment: str):
    import ast
    try:
        segment = segment.encode('utf-8')
    except UnicodeEncodeError:
        return None
    try:
        tree = ast.parse(segment)
    except SyntaxError:
        return None
    if len(tree.body) != 1:
        return None
    statement = tree.body[0]
    if isinstance(statement, ast.Expr):
        expr = statement.value
        if isinstance(expr, ast.BinOp):
            operator_str = segment[expr.left.end_col_offset:expr.right.col_offset]
            operator_offset = len(operator_str) - len(operator_str.lstrip())
            left_anchor = expr.left.end_col_offset + operator_offset
            right_anchor = left_anchor + 1
            if operator_offset + 1 < len(operator_str) and (not operator_str[operator_offset + 1] == ord(b' ')):
                right_anchor += 1
            return (left_anchor, right_anchor)
        if isinstance(expr, ast.Subscript):
            return (expr.value.end_col_offset, expr.slice.end_col_offset + 1)
    return None