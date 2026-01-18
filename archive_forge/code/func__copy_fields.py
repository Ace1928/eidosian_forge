import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def _copy_fields(ary):
    """Return copy of structured array with padding between fields removed.

    Parameters
    ----------
    ary : ndarray
       Structured array from which to remove padding bytes

    Returns
    -------
    ary_copy : ndarray
       Copy of ary with padding bytes removed
    """
    dt = ary.dtype
    copy_dtype = {'names': dt.names, 'formats': [dt.fields[name][0] for name in dt.names]}
    return array(ary, dtype=copy_dtype, copy=True)