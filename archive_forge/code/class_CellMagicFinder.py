import ast
import collections
import dataclasses
import secrets
import sys
from functools import lru_cache
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple
from black.output import out
from black.report import NothingChanged
class CellMagicFinder(ast.NodeVisitor):
    """Find cell magics.

    Note that the source of the abstract syntax tree
    will already have been processed by IPython's
    TransformerManager().transform_cell.

    For example,

        %%time

        foo()

    would have been transformed to

        get_ipython().run_cell_magic('time', '', 'foo()\\n')

    and we look for instances of the latter.
    """

    def __init__(self, cell_magic: Optional[CellMagic]=None) -> None:
        self.cell_magic = cell_magic

    def visit_Expr(self, node: ast.Expr) -> None:
        """Find cell magic, extract header and body."""
        if isinstance(node.value, ast.Call) and _is_ipython_magic(node.value.func) and (node.value.func.attr == 'run_cell_magic'):
            args = _get_str_args(node.value.args)
            self.cell_magic = CellMagic(name=args[0], params=args[1], body=args[2])
        self.generic_visit(node)