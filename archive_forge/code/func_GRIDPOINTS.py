from __future__ import annotations
import operator
import sys
from enum import IntEnum, IntFlag
from functools import reduce
from typing import Any, Literal, SupportsFloat, SupportsInt, Union
from . import Image, __version__
from ._deprecate import deprecate
from ._typing import SupportsRead
@staticmethod
def GRIDPOINTS(n: int) -> Flags:
    """
        Fine-tune control over number of gridpoints

        :param n: :py:class:`int` in range ``0 <= n <= 255``
        """
    return Flags.NONE | (n & 255) << 16