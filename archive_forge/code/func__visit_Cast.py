import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_Cast(self, node: ast.Cast):
    self.visit(node.type)
    self.stream.write('(')
    self.visit(node.operand)
    self.stream.write(')')