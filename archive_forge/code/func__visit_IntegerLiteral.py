import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_IntegerLiteral(self, node: ast.IntegerLiteral) -> None:
    self.stream.write(str(node.value))