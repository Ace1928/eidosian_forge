from __future__ import annotations
import ast
from typing import Any, Final
from streamlit import config
def _is_docstring_node(node, node_index, parent_type) -> bool:
    return node_index == 0 and _is_string_constant_node(node) and (parent_type in {ast.FunctionDef, ast.AsyncFunctionDef, ast.Module})