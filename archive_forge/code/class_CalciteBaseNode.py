import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr
class CalciteBaseNode(abc.ABC):
    """
    A base class for a Calcite computation sequence node.

    Calcite nodes are not combined into a tree but usually stored
    in a sequence which works similar to a stack machine: the result
    of the previous operation is an implicit operand of the current
    one. Input nodes also can be referenced directly via its unique
    ID number.

    Calcite nodes structure is based on a JSON representation used by
    HDK for parsed queries serialization/deserialization for
    interactions with a Calcite server. Currently, this format is
    internal and is not a part of public API. It's not documented
    and can be modified in an incompatible way in the future.

    Parameters
    ----------
    relOp : str
        An operation name.

    Attributes
    ----------
    id : int
        Id of the node. Should be unique within a single query.
    relOp : str
        Operation name.
    """
    _next_id = [0]

    def __init__(self, relOp):
        self.id = str(type(self)._next_id[0])
        type(self)._next_id[0] += 1
        self.relOp = relOp

    @classmethod
    def reset_id(cls, next_id=0):
        """
        Reset ID to be used for the next new node to `next_id`.

        Can be used to have a zero-based numbering for each
        generated query.

        Parameters
        ----------
        next_id : int, default: 0
            Next node id.
        """
        cls._next_id[0] = next_id