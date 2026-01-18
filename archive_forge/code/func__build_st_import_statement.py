from __future__ import annotations
import ast
from typing import Any, Final
from streamlit import config
def _build_st_import_statement():
    """Build AST node for `import magic_funcs as __streamlitmagic__`."""
    return ast.Import(names=[ast.alias(name='streamlit.runtime.scriptrunner.magic_funcs', asname=MAGIC_MODULE_NAME)])