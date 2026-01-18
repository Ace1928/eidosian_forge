from __future__ import annotations
from functools import wraps
from typing import (
from pandas._libs.lib import item_from_zerodim
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.generic import (
def _unpack_zerodim_and_defer(method, name: str):
    """
    Boilerplate for pandas conventions in arithmetic and comparison methods.

    Ensure method returns NotImplemented when operating against "senior"
    classes.  Ensure zero-dimensional ndarrays are always unpacked.

    Parameters
    ----------
    method : binary method
    name : str

    Returns
    -------
    method
    """
    stripped_name = name.removeprefix('__').removesuffix('__')
    is_cmp = stripped_name in {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}

    @wraps(method)
    def new_method(self, other):
        if is_cmp and isinstance(self, ABCIndex) and isinstance(other, ABCSeries):
            pass
        else:
            prio = getattr(other, '__pandas_priority__', None)
            if prio is not None:
                if prio > self.__pandas_priority__:
                    return NotImplemented
        other = item_from_zerodim(other)
        return method(self, other)
    return new_method