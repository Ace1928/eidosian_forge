import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_Binary(self, node: ast.Binary):
    if isinstance(node.left, (ast.Unary, ast.Binary)) and _BINDING_POWER[node.left.op].right < _BINDING_POWER[node.op].left:
        self.stream.write('(')
        self.visit(node.left)
        self.stream.write(')')
    else:
        self.visit(node.left)
    self.stream.write(f' {node.op.value} ')
    if isinstance(node.right, (ast.Unary, ast.Binary)) and _BINDING_POWER[node.right.op].left < _BINDING_POWER[node.op].right:
        self.stream.write('(')
        self.visit(node.right)
        self.stream.write(')')
    else:
        self.visit(node.right)