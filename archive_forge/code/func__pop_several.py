from __future__ import annotations
import math
from collections import OrderedDict
from typing import TYPE_CHECKING
import attrs
from .. import _core
from .._util import final
def _pop_several(self, count: int | float) -> Iterator[Task]:
    if isinstance(count, float):
        if math.isinf(count):
            count = len(self._parked)
        else:
            raise ValueError('Cannot pop a non-integer number of tasks.')
    else:
        count = min(count, len(self._parked))
    for _ in range(count):
        task, _ = self._parked.popitem(last=False)
        yield task