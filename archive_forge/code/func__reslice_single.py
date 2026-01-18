import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
def _reslice_single(self, batch: T) -> Iterable[Tuple[bool, T]]:
    a = self.get_keys_ndarray(batch, self._keys)
    b = np.roll(a, 1, axis=0)
    diff = self._diff(a, b)
    if self._last_row is not None:
        diff_from_last: bool = self._diff(a[0:1], self._last_row)[0]
    else:
        diff_from_last = True
    self._last_row = a[-1:]
    points = np.where(diff)[0].tolist() + [a.shape[0]]
    if len(points) == 1:
        yield (diff_from_last, batch)
    else:
        for i in range(len(points) - 1):
            new_start = diff_from_last if i == 0 else True
            yield (new_start, self.take(batch, points[i], points[i + 1] - points[i]))