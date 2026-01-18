from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
class _Tile(NamedTuple):
    encoder_name: str
    extents: tuple[int, int, int, int]
    offset: int
    args: tuple[Any, ...] | str | None