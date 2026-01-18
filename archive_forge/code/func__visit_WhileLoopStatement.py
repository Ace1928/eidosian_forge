import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_WhileLoopStatement(self, node: ast.WhileLoopStatement) -> None:
    self._start_line()
    self.stream.write('while (')
    self.visit(node.condition)
    self.stream.write(') ')
    self.visit(node.body)
    self._end_line()