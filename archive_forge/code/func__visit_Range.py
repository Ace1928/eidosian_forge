import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_Range(self, node: ast.Range) -> None:
    if node.start is not None:
        self.visit(node.start)
    self.stream.write(':')
    if node.step is not None:
        self.visit(node.step)
        self.stream.write(':')
    if node.end is not None:
        self.visit(node.end)