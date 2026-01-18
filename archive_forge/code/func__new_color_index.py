from __future__ import annotations
import array
from . import GimpGradientFile, GimpPaletteFile, ImageColor, PaletteFile
def _new_color_index(self, image=None, e=None):
    if not isinstance(self.palette, bytearray):
        self._palette = bytearray(self.palette)
    index = len(self.palette) // 3
    special_colors = ()
    if image:
        special_colors = (image.info.get('background'), image.info.get('transparency'))
        while index in special_colors:
            index += 1
    if index >= 256:
        if image:
            for i, count in reversed(list(enumerate(image.histogram()))):
                if count == 0 and i not in special_colors:
                    index = i
                    break
        if index >= 256:
            msg = 'cannot allocate more than 256 colors'
            raise ValueError(msg) from e
    return index