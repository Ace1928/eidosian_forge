from __future__ import annotations
import io
from typing import BinaryIO, Callable
from . import FontFile, Image
from ._binary import i8
from ._binary import i16be as b16
from ._binary import i16le as l16
from ._binary import i32be as b32
from ._binary import i32le as l32
def _load_encoding(self) -> list[int | None]:
    fp, format, i16, i32 = self._getformat(PCF_BDF_ENCODINGS)
    first_col, last_col = (i16(fp.read(2)), i16(fp.read(2)))
    first_row, last_row = (i16(fp.read(2)), i16(fp.read(2)))
    i16(fp.read(2))
    nencoding = (last_col - first_col + 1) * (last_row - first_row + 1)
    encoding: list[int | None] = [None] * min(256, nencoding)
    encoding_offsets = [i16(fp.read(2)) for _ in range(nencoding)]
    for i in range(first_col, len(encoding)):
        try:
            encoding_offset = encoding_offsets[ord(bytearray([i]).decode(self.charset_encoding))]
            if encoding_offset != 65535:
                encoding[i] = encoding_offset
        except UnicodeDecodeError:
            pass
    return encoding