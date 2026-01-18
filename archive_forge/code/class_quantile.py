from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class quantile(Reducer):
    """
    Return the value for the nth percentile within the range of values for the
    field within the group.
    """
    NAME = 'QUANTILE'

    def __init__(self, field: str, pct: float) -> None:
        super().__init__(field, str(pct))
        self._field = field