import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_sequence(self, nodes: Sequence[ast.ASTNode], *, start: str='', end: str='', separator: str) -> None:
    if start:
        self.stream.write(start)
    for node in nodes[:-1]:
        self.visit(node)
        self.stream.write(separator)
    if nodes:
        self.visit(nodes[-1])
    if end:
        self.stream.write(end)