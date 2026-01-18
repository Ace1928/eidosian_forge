import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _write_statement(self, line: str) -> None:
    self._start_line()
    self.stream.write(line)
    self._end_statement()