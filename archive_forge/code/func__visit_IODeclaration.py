import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_IODeclaration(self, node: ast.IODeclaration) -> None:
    self._start_line()
    modifier = 'input' if node.modifier is ast.IOModifier.INPUT else 'output'
    self.stream.write(modifier + ' ')
    self.visit(node.type)
    self.stream.write(' ')
    self.visit(node.identifier)
    self._end_statement()