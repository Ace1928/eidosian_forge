from __future__ import annotations
import io
from typing import BinaryIO, Callable
from . import FontFile, Image
from ._binary import i8
from ._binary import i16be as b16
from ._binary import i16le as l16
from ._binary import i32be as b32
from ._binary import i32le as l32
def _load_bitmaps(self, metrics: list[tuple[int, int, int, int, int, int, int, int]]) -> list[Image.Image]:
    fp, format, i16, i32 = self._getformat(PCF_BITMAPS)
    nbitmaps = i32(fp.read(4))
    if nbitmaps != len(metrics):
        msg = 'Wrong number of bitmaps'
        raise OSError(msg)
    offsets = [i32(fp.read(4)) for _ in range(nbitmaps)]
    bitmap_sizes = [i32(fp.read(4)) for _ in range(4)]
    bitorder = format & 8
    padindex = format & 3
    bitmapsize = bitmap_sizes[padindex]
    offsets.append(bitmapsize)
    data = fp.read(bitmapsize)
    pad = BYTES_PER_ROW[padindex]
    mode = '1;R'
    if bitorder:
        mode = '1'
    bitmaps = []
    for i in range(nbitmaps):
        xsize, ysize = metrics[i][:2]
        b, e = offsets[i:i + 2]
        bitmaps.append(Image.frombytes('1', (xsize, ysize), data[b:e], 'raw', mode, pad(xsize)))
    return bitmaps