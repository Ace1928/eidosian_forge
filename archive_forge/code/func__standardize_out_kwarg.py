from __future__ import annotations
import operator
from typing import Any
import numpy as np
from pandas._libs import lib
from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op
from pandas.core.dtypes.generic import ABCNDFrame
from pandas.core import roperator
from pandas.core.construction import extract_array
from pandas.core.ops.common import unpack_zerodim_and_defer
def _standardize_out_kwarg(**kwargs) -> dict:
    """
    If kwargs contain "out1" and "out2", replace that with a tuple "out"

    np.divmod, np.modf, np.frexp can have either `out=(out1, out2)` or
    `out1=out1, out2=out2)`
    """
    if 'out' not in kwargs and 'out1' in kwargs and ('out2' in kwargs):
        out1 = kwargs.pop('out1')
        out2 = kwargs.pop('out2')
        out = (out1, out2)
        kwargs['out'] = out
    return kwargs