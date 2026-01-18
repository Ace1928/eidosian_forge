import abc
from collections.abc import Mapping
from typing import TypeVar, Generic
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
def _toposort_graph(self, scfg: SCFG):
    toposorted = toposort_graph(scfg.graph)
    if self.direction == 'forward':
        return toposorted
    elif self.direction == 'backward':
        return reversed(toposorted)
    else:
        assert False, f'invalid direction {self.direction!r}'