import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_ReturnStatement(self, node: ast.ReturnStatement) -> None:
    self._start_line()
    if node.expression:
        self.stream.write('return ')
        self.visit(node.expression)
    else:
        self.stream.write('return')
    self._end_statement()