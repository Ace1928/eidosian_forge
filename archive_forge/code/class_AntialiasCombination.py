from __future__ import annotations
from enum import Enum
from typing import NamedTuple, TYPE_CHECKING
class AntialiasCombination(Enum):
    SUM_1AGG = 1
    SUM_2AGG = 2
    MIN = 3
    MAX = 4
    FIRST = 5
    LAST = 6