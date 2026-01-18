import array
from typing import (
def _parse_horiz2(self, n: Any) -> BitParserState:
    if n is None:
        raise self.InvalidData
    self._n2 += n
    if n < 64:
        self._color = 1 - self._color
        self._accept = self._parse_mode
        self._do_horizontal(self._n1, self._n2)
        self._flush_line()
        return self.MODE
    elif self._color:
        return self.WHITE
    else:
        return self.BLACK