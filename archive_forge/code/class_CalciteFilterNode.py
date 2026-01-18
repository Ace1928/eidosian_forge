import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr
class CalciteFilterNode(CalciteBaseNode):
    """
    A node to represent a filter operation.

    Parameters
    ----------
    condition : BaseExpr
        A filtering condition.

    Attributes
    ----------
    condition : BaseExpr
        A filter to apply.
    """

    def __init__(self, condition):
        super(CalciteFilterNode, self).__init__('LogicalFilter')
        self.condition = condition