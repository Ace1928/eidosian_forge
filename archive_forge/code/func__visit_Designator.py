import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_Designator(self, node: ast.Designator) -> None:
    self.stream.write('[')
    self.visit(node.expression)
    self.stream.write(']')