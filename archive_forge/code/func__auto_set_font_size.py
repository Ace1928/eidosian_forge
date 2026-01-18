import numpy as np
from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path
def _auto_set_font_size(self, renderer):
    if len(self._cells) == 0:
        return
    fontsize = next(iter(self._cells.values())).get_fontsize()
    cells = []
    for key, cell in self._cells.items():
        if key[1] in self._autoColumns:
            continue
        size = cell.auto_set_font_size(renderer)
        fontsize = min(fontsize, size)
        cells.append(cell)
    for cell in self._cells.values():
        cell.set_fontsize(fontsize)