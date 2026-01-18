import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _end_line(self) -> None:
    self.stream.write('\n')