import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_SubscriptedIdentifier(self, node: ast.SubscriptedIdentifier) -> None:
    self.stream.write(node.string)
    self.stream.write('[')
    self.visit(node.subscript)
    self.stream.write(']')