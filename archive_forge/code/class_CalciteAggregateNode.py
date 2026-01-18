import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr
class CalciteAggregateNode(CalciteBaseNode):
    """
    A node to represent an aggregate operation.

    Parameters
    ----------
    fields : list of str
        Output field names.
    group : list of CalciteInputIdxExpr
        Group key columns.
    aggs : list of BaseExpr
        Aggregates to compute.

    Attributes
    ----------
    fields : list of str
        Output field names.
    group : list of CalciteInputIdxExpr
        Group key columns.
    aggs : list of BaseExpr
        Aggregates to compute.
    """

    def __init__(self, fields, group, aggs):
        super(CalciteAggregateNode, self).__init__('LogicalAggregate')
        self.fields = [ColNameCodec.encode(field) for field in fields]
        self.group = group
        self.aggs = aggs