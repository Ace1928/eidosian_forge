import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_Pragma(self, node: ast.Pragma) -> None:
    self._write_statement(f'#pragma {node.content}')