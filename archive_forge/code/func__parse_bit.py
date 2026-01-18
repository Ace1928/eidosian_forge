import array
from typing import (
def _parse_bit(self, x: object) -> None:
    if x:
        v = self._state[1]
    else:
        v = self._state[0]
    self._pos += 1
    if isinstance(v, list):
        self._state = v
    else:
        assert self._accept is not None
        self._state = self._accept(v)