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
def _get_phi_name(self, varname: str, label: str) -> str:
    suffix = str(self._get_label(label))
    return f'$phi.{varname}.{suffix}'