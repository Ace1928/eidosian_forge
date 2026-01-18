from __future__ import annotations
import olefile
from . import Image, ImageFile
from ._binary import i32le as i32
def _open_index(self, index=1):
    prop = self.ole.getproperties([f'Data Object Store {index:06d}', '\x05Image Contents'])
    self._size = (prop[16777218], prop[16777219])
    size = max(self.size)
    i = 1
    while size > 64:
        size = size / 2
        i += 1
    self.maxid = i - 1
    id = self.maxid << 16
    s = prop[33554434 | id]
    bands = i32(s, 4)
    if bands > 4:
        msg = 'Invalid number of bands'
        raise OSError(msg)
    colors = tuple((i32(s, 8 + i * 4) & 2147483647 for i in range(bands)))
    self._mode, self.rawmode = MODES[colors]
    self.jpeg = {}
    for i in range(256):
        id = 50331649 | i << 16
        if id in prop:
            self.jpeg[i] = prop[id]
    self._open_subimage(1, self.maxid)