from __future__ import annotations
import ast
from typing import Any, Final
from streamlit import config
def _insert_import_statement(tree: Any) -> None:
    """Insert Streamlit import statement at the top(ish) of the tree."""
    st_import = _build_st_import_statement()
    if tree.body and type(tree.body[0]) in {ast.ImportFrom, ast.Import}:
        tree.body.insert(1, st_import)
    elif len(tree.body) > 1 and (type(tree.body[0]) is ast.Expr and _is_string_constant_node(tree.body[0].value)) and (type(tree.body[1]) in {ast.ImportFrom, ast.Import}):
        tree.body.insert(2, st_import)
    else:
        tree.body.insert(0, st_import)