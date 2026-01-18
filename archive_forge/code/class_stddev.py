from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class stddev(FieldOnlyReducer):
    """
    Return the standard deviation for the values within the group
    """
    NAME = 'STDDEV'

    def __init__(self, field: str) -> None:
        super().__init__(field)