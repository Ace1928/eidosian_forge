import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_StringifyAndPray(self, node: ast.StringifyAndPray) -> None:
    self.stream.write(str(node.obj))