from __future__ import annotations
from enum import Enum
from typing import NamedTuple, TYPE_CHECKING
class AntialiasStage2(NamedTuple):
    """Configuration for second-stage combination of a single antialiased reduction."""
    combination: AntialiasCombination
    zero: float
    n_reduction: bool = False
    categorical: bool = False