import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumReset(self, node: ast.QuantumReset) -> None:
    self._start_line()
    self.stream.write('reset ')
    self.visit(node.identifier)
    self._end_statement()