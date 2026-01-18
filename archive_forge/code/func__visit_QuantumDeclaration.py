import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumDeclaration(self, node: ast.QuantumDeclaration) -> None:
    self._start_line()
    self.stream.write('qubit')
    if node.designator is not None:
        self.visit(node.designator)
    self.stream.write(' ')
    self.visit(node.identifier)
    self._end_statement()