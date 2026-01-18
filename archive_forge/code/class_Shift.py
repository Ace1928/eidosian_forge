from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable, Optional, Union, cast
import numpy as np
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._core.typing import Default
@dataclass
class Shift(Move):
    """
    Displacement of all marks with the same magnitude / direction.

    Parameters
    ----------
    x, y : float
        Magnitude of shift, in data units, along each axis.

    Examples
    --------
    .. include:: ../docstrings/objects.Shift.rst

    """
    x: float = 0
    y: float = 0

    def __call__(self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale]) -> DataFrame:
        data = data.copy(deep=False)
        data['x'] = data['x'] + self.x
        data['y'] = data['y'] + self.y
        return data