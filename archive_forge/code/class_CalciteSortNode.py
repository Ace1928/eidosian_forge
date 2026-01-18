import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr
class CalciteSortNode(CalciteBaseNode):
    """
    A node to represent a sort operation.

    Parameters
    ----------
    collation : list of CalciteCollation
        Sort keys.

    Attributes
    ----------
    collation : list of CalciteCollation
        Sort keys.
    """

    def __init__(self, collation):
        super(CalciteSortNode, self).__init__('LogicalSort')
        self.collation = collation