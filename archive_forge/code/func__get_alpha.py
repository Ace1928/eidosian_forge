from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Generic
import numpy as np
from packaging.version import Version
from xarray.core.computation import apply_ufunc
from xarray.core.options import _get_keep_attrs
from xarray.core.pdcompat import count_not_none
from xarray.core.types import T_DataWithCoords
from xarray.core.utils import module_available
from xarray.namedarray import pycompat
def _get_alpha(com: float | None=None, span: float | None=None, halflife: float | None=None, alpha: float | None=None) -> float:
    """
    Convert com, span, halflife to alpha.
    """
    valid_count = count_not_none(com, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError('com, span, halflife, and alpha are mutually exclusive')
    if com is not None:
        if com < 0:
            raise ValueError('commust satisfy: com>= 0')
        return 1 / (com + 1)
    elif span is not None:
        if span < 1:
            raise ValueError('span must satisfy: span >= 1')
        return 2 / (span + 1)
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError('halflife must satisfy: halflife > 0')
        return 1 - np.exp(np.log(0.5) / halflife)
    elif alpha is not None:
        if not 0 < alpha <= 1:
            raise ValueError('alpha must satisfy: 0 < alpha <= 1')
        return alpha
    else:
        raise ValueError('Must pass one of comass, span, halflife, or alpha')