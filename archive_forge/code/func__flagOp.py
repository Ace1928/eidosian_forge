from __future__ import division, absolute_import
from functools import partial
from itertools import count
from operator import and_, or_, xor
def _flagOp(op, left, right):
    """
    Implement a binary operator for a L{FlagConstant} instance.

    @param op: A two-argument callable implementing the binary operation.  For
        example, C{operator.or_}.

    @param left: The left-hand L{FlagConstant} instance.
    @param right: The right-hand L{FlagConstant} instance.

    @return: A new L{FlagConstant} instance representing the result of the
        operation.
    """
    value = op(left.value, right.value)
    names = op(left.names, right.names)
    result = FlagConstant()
    result._realize(left._container, names, value)
    return result