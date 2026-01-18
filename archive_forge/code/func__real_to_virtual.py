import time
from math import inf
from .. import _core
from .._abc import Clock
from .._util import final
from ._run import GLOBAL_RUN_CONTEXT
def _real_to_virtual(self, real: float) -> float:
    real_offset = real - self._real_base
    virtual_offset = self._rate * real_offset
    return self._virtual_base + virtual_offset