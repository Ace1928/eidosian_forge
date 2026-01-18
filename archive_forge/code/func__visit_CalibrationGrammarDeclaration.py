import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_CalibrationGrammarDeclaration(self, node: ast.CalibrationGrammarDeclaration) -> None:
    self._write_statement(f'defcalgrammar "{node.name}"')