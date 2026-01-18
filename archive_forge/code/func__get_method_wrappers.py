from __future__ import annotations
import operator
from pandas.core.dtypes.generic import (
from pandas.core.ops import roperator
def _get_method_wrappers(cls):
    """
    Find the appropriate operation-wrappers to use when defining flex/special
    arithmetic, boolean, and comparison operations with the given class.

    Parameters
    ----------
    cls : class

    Returns
    -------
    arith_flex : function or None
    comp_flex : function or None
    """
    from pandas.core.ops import flex_arith_method_FRAME, flex_comp_method_FRAME, flex_method_SERIES
    if issubclass(cls, ABCSeries):
        arith_flex = flex_method_SERIES
        comp_flex = flex_method_SERIES
    elif issubclass(cls, ABCDataFrame):
        arith_flex = flex_arith_method_FRAME
        comp_flex = flex_comp_method_FRAME
    return (arith_flex, comp_flex)