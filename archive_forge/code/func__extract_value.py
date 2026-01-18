import ast
import sys
from typing import Any
from .internal import Filters, Key
def _extract_value(node) -> Any:
    if sys.version_info < (3, 8) and isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Constant):
        return node.n
    if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        return [_extract_value(element) for element in node.elts]
    if isinstance(node, ast.Name):
        return node.id
    raise ValueError(f'Unsupported value type: {type(node)}')