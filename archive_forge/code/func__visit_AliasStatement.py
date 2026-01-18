import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_AliasStatement(self, node: ast.AliasStatement) -> None:
    self._start_line()
    self.stream.write('let ')
    self.visit(node.identifier)
    self.stream.write(' = ')
    self.visit(node.value)
    self._end_statement()