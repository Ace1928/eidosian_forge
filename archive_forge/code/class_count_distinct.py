from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class count_distinct(FieldOnlyReducer):
    """
    Calculate the number of distinct values contained in all the results in
    the group for the given field
    """
    NAME = 'COUNT_DISTINCT'

    def __init__(self, field: str) -> None:
        super().__init__(field)