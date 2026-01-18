from __future__ import annotations
from io import BytesIO
from typing import Sequence
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._deprecate import deprecate
def _i8(c: int | bytes) -> int:
    return c if isinstance(c, int) else c[0]