import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_ContinueStatement(self, _node: ast.ContinueStatement) -> None:
    self._write_statement('continue')