from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class min(FieldOnlyReducer):
    """
    Calculates the smallest value in the given field within the group
    """
    NAME = 'MIN'

    def __init__(self, field: str) -> None:
        super().__init__(field)