from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
class RowStringIterator(RowStringConverter):
    """Iterator over rows of the header or the body of the table."""

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iterate over LaTeX string representations of rows."""