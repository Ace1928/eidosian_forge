import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def _getrange(self, bbox: Rect) -> Iterator[Point]:
    x0, y0, x1, y1 = bbox
    if x1 <= self.x0 or self.x1 <= x0 or y1 <= self.y0 or (self.y1 <= y0):
        return
    x0 = max(self.x0, x0)
    y0 = max(self.y0, y0)
    x1 = min(self.x1, x1)
    y1 = min(self.y1, y1)
    for grid_y in drange(y0, y1, self.gridsize):
        for grid_x in drange(x0, x1, self.gridsize):
            yield (grid_x, grid_y)