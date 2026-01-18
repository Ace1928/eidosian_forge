import math
from collections import deque
from typing import Any, Dict, List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.util import locality_string
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.types import ObjectRef
def _split_from_buffer(self, nrow: int) -> List[RefBundle]:
    output = []
    acc = 0
    while acc < nrow:
        b = self._buffer.pop()
        if acc + b.num_rows() <= nrow:
            output.append(b)
            acc += b.num_rows()
        else:
            left, right = _split(b, nrow - acc)
            output.append(left)
            acc += left.num_rows()
            self._buffer.append(right)
            assert acc == nrow, (acc, nrow)
    assert sum((b.num_rows() for b in output)) == nrow, (acc, nrow)
    return output