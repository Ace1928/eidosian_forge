from __future__ import annotations
import ast
from typing import Any, Final
from streamlit import config
def _is_displayable_last_expr(is_root: bool, is_last_expr: bool, file_ends_in_semicolon: bool) -> bool:
    return is_last_expr and is_root and (not file_ends_in_semicolon) and config.get_option('magic.displayLastExprIfNoSemicolon')