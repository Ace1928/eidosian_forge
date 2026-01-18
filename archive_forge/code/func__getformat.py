from __future__ import annotations
import io
from typing import BinaryIO, Callable
from . import FontFile, Image
from ._binary import i8
from ._binary import i16be as b16
from ._binary import i16le as l16
from ._binary import i32be as b32
from ._binary import i32le as l32
def _getformat(self, tag: int) -> tuple[BinaryIO, int, Callable[[bytes], int], Callable[[bytes], int]]:
    format, size, offset = self.toc[tag]
    fp = self.fp
    fp.seek(offset)
    format = l32(fp.read(4))
    if format & 4:
        i16, i32 = (b16, b32)
    else:
        i16, i32 = (l16, l32)
    return (fp, format, i16, i32)