import array
from typing import (
def _flush_line(self) -> None:
    if self.width <= self._curpos:
        self.output_line(self._y, self._curline)
        self._y += 1
        self._reset_line()
        if self.bytealign:
            raise self.ByteSkip
    return