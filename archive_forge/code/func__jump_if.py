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
def _jump_if(self, pred):
    """Emit code for jump if predicate is true."""
    self.branch_predicate = self.store(self.vsmap[pred], '$jump_if')