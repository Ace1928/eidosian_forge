import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def _newnames(datatype, order):
    """
    Given a datatype and an order object, return a new names tuple, with the
    order indicated
    """
    oldnames = datatype.names
    nameslist = list(oldnames)
    if isinstance(order, str):
        order = [order]
    seen = set()
    if isinstance(order, (list, tuple)):
        for name in order:
            try:
                nameslist.remove(name)
            except ValueError:
                if name in seen:
                    raise ValueError(f'duplicate field name: {name}') from None
                else:
                    raise ValueError(f'unknown field name: {name}') from None
            seen.add(name)
        return tuple(list(order) + nameslist)
    raise ValueError(f'unsupported order value: {order}')