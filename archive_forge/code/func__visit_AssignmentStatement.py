import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_AssignmentStatement(self, node: ast.AssignmentStatement) -> None:
    self._start_line()
    self.visit(node.lvalue)
    self.stream.write(' = ')
    self.visit(node.rvalue)
    self._end_statement()