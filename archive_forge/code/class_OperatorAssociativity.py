import enum
from contextlib import nullcontext
from pyomo.common.deprecation import deprecated
class OperatorAssociativity(enum.IntEnum):
    """Enum for indicating the associativity of an operator.

    LEFT_TO_RIGHT(1) if this operator is left-to-right associative or
    RIGHT_TO_LEFT(-1) if it is right-to-left associative.  Any other
    values will be interpreted as "not associative" (implying any
    arguments that are at this operator's PRECEDENCE will be enclosed
    in parens).

    """
    RIGHT_TO_LEFT = -1
    NON_ASSOCIATIVE = 0
    LEFT_TO_RIGHT = 1