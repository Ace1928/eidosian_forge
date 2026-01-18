import math
import warnings
import numpy as np
import dataclasses
from typing import Optional, Callable
from functools import partial
from scipy._lib._util import _asarray_validated
from . import _distance_wrap
from . import _hausdorff
from ..linalg import norm
from ..special import rel_entr
from . import _distance_pybind
def _validate_cdist_input(XA, XB, mA, mB, n, metric_info, **kwargs):
    types = metric_info.types
    typ = types[types.index(XA.dtype)] if XA.dtype in types else types[0]
    XA = _convert_to_type(XA, out_type=typ)
    XB = _convert_to_type(XB, out_type=typ)
    _validate_kwargs = metric_info.validator
    if _validate_kwargs:
        kwargs = _validate_kwargs((XA, XB), mA + mB, n, **kwargs)
    return (XA, XB, typ, kwargs)