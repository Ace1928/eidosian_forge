import array
from typing import (
def _reset_line(self) -> None:
    self._refline = self._curline
    self._curline = array.array('b', [1] * self.width)
    self._curpos = -1
    self._color = 1
    return