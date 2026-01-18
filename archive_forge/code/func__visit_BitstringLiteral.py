import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_BitstringLiteral(self, node: ast.BitstringLiteral):
    self.stream.write(f'"{node.value:0{node.width}b}"')