from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
class Direction(IntEnum):
    INPUT = 0
    OUTPUT = 1
    PROOF = 2