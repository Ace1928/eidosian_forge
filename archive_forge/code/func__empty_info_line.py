from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@property
def _empty_info_line(self) -> str:
    return f'Empty {type(self.frame).__name__}\nColumns: {self.frame.columns}\nIndex: {self.frame.index}'