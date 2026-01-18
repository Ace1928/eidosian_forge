import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumMeasurementAssignment(self, node: ast.QuantumMeasurementAssignment) -> None:
    self._start_line()
    self.visit(node.identifier)
    self.stream.write(' = ')
    self.visit(node.quantumMeasurement)
    self._end_statement()