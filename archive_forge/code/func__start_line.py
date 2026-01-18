import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _start_line(self) -> None:
    self.stream.write(self._current_indent * self.indent)