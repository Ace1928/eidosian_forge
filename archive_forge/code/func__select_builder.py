from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _select_builder(self) -> type[TableBuilderAbstract]:
    """Select proper table builder."""
    if self.longtable:
        return LongTableBuilder
    if any([self.caption, self.label, self.position]):
        return RegularTableBuilder
    return TabularBuilder