import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_Constant(self, node: ast.Constant) -> None:
    self.stream.write(self._CONSTANT_LOOKUP[node])