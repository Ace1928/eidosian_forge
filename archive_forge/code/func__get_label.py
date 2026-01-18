import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
def _get_label(self, label: str) -> int:
    num = self._label_map.setdefault(f'block.{label}', len(self._label_map))
    return num