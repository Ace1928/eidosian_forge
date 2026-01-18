import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr
class CalciteJoinNode(CalciteBaseNode):
    """
    A node to represent a join operation.

    Parameters
    ----------
    left_id : int
        ID of the left join operand.
    right_id : int
        ID of the right join operand.
    how : str
        Type of the join.
    condition : BaseExpr
        Join condition.

    Attributes
    ----------
    inputs : list of int
        IDs of the left and the right operands of the join.
    joinType : str
        Type of the join.
    condition : BaseExpr
        Join condition.
    """

    def __init__(self, left_id, right_id, how, condition):
        super(CalciteJoinNode, self).__init__('LogicalJoin')
        self.inputs = [left_id, right_id]
        self.joinType = how
        self.condition = condition