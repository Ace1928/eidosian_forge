import array
from typing import (
def _parse_horiz1(self, n: Any) -> BitParserState:
    if n is None:
        raise self.InvalidData
    self._n1 += n
    if n < 64:
        self._n2 = 0
        self._color = 1 - self._color
        self._accept = self._parse_horiz2
    if self._color:
        return self.WHITE
    else:
        return self.BLACK