from __future__ import annotations
import ast
from typing import Any, Final
from streamlit import config
def _should_display_docstring_like_node_anyway(is_root: bool) -> bool:
    return config.get_option('magic.displayRootDocString') and is_root