import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_Header(self, node: ast.Header) -> None:
    self.visit(node.version)
    for include in node.includes:
        self.visit(include)