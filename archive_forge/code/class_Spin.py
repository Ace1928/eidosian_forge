from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
@unique
class Spin(Enum):
    """Enum type for Spin. Only up and down. Usage: Spin.up, Spin.down."""
    up, down = (1, -1)

    def __int__(self) -> int:
        return self.value

    def __float__(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return str(self.value)