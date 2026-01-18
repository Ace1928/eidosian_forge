from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@property
def _position_macro(self) -> str:
    """Position macro, extracted from self.position, like [h]."""
    return f'[{self.position}]' if self.position else ''