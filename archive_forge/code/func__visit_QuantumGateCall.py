import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumGateCall(self, node: ast.QuantumGateCall) -> None:
    self._start_line()
    if node.modifiers:
        self._visit_sequence(node.modifiers, end=' @ ', separator=' @ ')
    self.visit(node.quantumGateName)
    if node.parameters:
        self._visit_sequence(node.parameters, start='(', end=')', separator=', ')
    self.stream.write(' ')
    self._visit_sequence(node.indexIdentifierList, separator=', ')
    self._end_statement()