import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr
class CalciteCollation:
    """
    A structure to describe sorting order.

    Parameters
    ----------
    field : CalciteInputIdxExpr
        A column to sort by.
    dir : {"ASCENDING", "DESCENDING"}, default: "ASCENDING"
        A sort order.
    nulls : {"LAST", "FIRST"}, default: "LAST"
        NULLs position after the sort.

    Attributes
    ----------
    field : CalciteInputIdxExpr
        A column to sort by.
    dir : {"ASCENDING", "DESCENDING"}
        A sort order.
    nulls : {"LAST", "FIRST"}
        NULLs position after the sort.
    """

    def __init__(self, field, dir='ASCENDING', nulls='LAST'):
        self.field = field
        self.direction = dir
        self.nulls = nulls