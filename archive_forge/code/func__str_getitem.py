from __future__ import annotations
import abc
from typing import (
import numpy as np
def _str_getitem(self, key):
    if isinstance(key, slice):
        return self._str_slice(start=key.start, stop=key.stop, step=key.step)
    else:
        return self._str_get(key)