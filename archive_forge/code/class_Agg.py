from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable
import pandas as pd
from pandas import DataFrame
from seaborn._core.scales import Scale
from seaborn._core.groupby import GroupBy
from seaborn._stats.base import Stat
from seaborn._statistics import (
from seaborn._core.typing import Vector
@dataclass
class Agg(Stat):
    """
    Aggregate data along the value axis using given method.

    Parameters
    ----------
    func : str or callable
        Name of a :class:`pandas.Series` method or a vector -> scalar function.

    See Also
    --------
    objects.Est : Aggregation with error bars.

    Examples
    --------
    .. include:: ../docstrings/objects.Agg.rst

    """
    func: str | Callable[[Vector], float] = 'mean'
    group_by_orient: ClassVar[bool] = True

    def __call__(self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale]) -> DataFrame:
        var = {'x': 'y', 'y': 'x'}.get(orient)
        res = groupby.agg(data, {var: self.func}).dropna(subset=[var]).reset_index(drop=True)
        return res