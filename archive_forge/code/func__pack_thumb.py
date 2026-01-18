import struct
from typing import Union
def _pack_thumb(self, val: int) -> bytes:
    b = bytes([val >> 11 & 255, 240 | val >> 19 & 7, val & 255, 248 | val >> 8 & 7])
    return b