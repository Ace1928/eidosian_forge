import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_CalibrationDefinition(self, node: ast.CalibrationDefinition) -> None:
    self._start_line()
    self.stream.write('defcal ')
    self.visit(node.name)
    self.stream.write(' ')
    if node.calibrationArgumentList:
        self._visit_sequence(node.calibrationArgumentList, start='(', end=')', separator=', ')
        self.stream.write(' ')
    self._visit_sequence(node.identifierList, separator=', ')
    self.stream.write(' {}')
    self._end_line()