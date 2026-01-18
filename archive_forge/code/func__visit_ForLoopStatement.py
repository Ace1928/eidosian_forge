import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_ForLoopStatement(self, node: ast.ForLoopStatement) -> None:
    self._start_line()
    self.stream.write('for ')
    self.visit(node.parameter)
    self.stream.write(' in ')
    if isinstance(node.indexset, ast.Range):
        self.stream.write('[')
        self.visit(node.indexset)
        self.stream.write(']')
    else:
        self.visit(node.indexset)
    self.stream.write(' ')
    self.visit(node.body)
    self._end_line()