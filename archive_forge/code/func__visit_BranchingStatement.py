import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_BranchingStatement(self, node: ast.BranchingStatement, chained: bool=False) -> None:
    if not chained:
        self._start_line()
    self.stream.write('if (')
    self.visit(node.condition)
    self.stream.write(') ')
    self.visit(node.true_body)
    if node.false_body is not None:
        self.stream.write(' else ')
        if self._chain_else_if and len(node.false_body.statements) == 1 and isinstance(node.false_body.statements[0], ast.BranchingStatement):
            self._visit_BranchingStatement(node.false_body.statements[0], chained=True)
        else:
            self.visit(node.false_body)
    if not chained:
        self._end_line()