import os
import sys
import hashlib
import uuid
import warnings
import collections
import weakref
import requests
import numpy as np
from .. import ndarray
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def _check_same_symbol_type(symbols):
    """Check whether all the symbols in the list are of the same type.
    Raise type error if the types are different. Return the class of
    the symbols."""
    from ..symbol.numpy import _Symbol as np_symbol
    from ..symbol import Symbol as nd_symbol
    is_np_sym = isinstance(symbols[0], np_symbol)
    for s in symbols[1:]:
        if is_np_sym != isinstance(s, np_symbol):
            raise TypeError('Found both classic symbol (mx.sym.Symbol) and numpy symbol (mx.sym.np._Symbol) in outputs. This will prevent you from building a computation graph by grouping them since different types of symbols are not allowed to be grouped in Gluon to form a computation graph. You will need to convert them to the same type of symbols, either classic or numpy following this rule: if you want numpy ndarray output(s) from the computation graph, please convert all the classic symbols in the list to numpy symbols by calling `as_np_ndarray()` on each of them; if you want classic ndarray output(s) from the computation graph, please convert all the numpy symbols in the list to classic symbols by calling `as_nd_ndarray()` on each of them.')
    return np_symbol if is_np_sym else nd_symbol