import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumBarrier(self, node: ast.QuantumBarrier) -> None:
    self._start_line()
    self.stream.write('barrier ')
    self._visit_sequence(node.indexIdentifierList, separator=', ')
    self._end_statement()