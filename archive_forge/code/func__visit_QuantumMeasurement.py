import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumMeasurement(self, node: ast.QuantumMeasurement) -> None:
    self.stream.write('measure ')
    self._visit_sequence(node.identifierList, separator=', ')