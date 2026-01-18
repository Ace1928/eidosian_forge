from __future__ import annotations
import functools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast, overload
import numpy as np
import pandas as pd
from xarray.core.alignment import broadcast
from xarray.core.concat import concat
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
def _center_pixels(x):
    """Center the pixels on the coordinates."""
    if np.issubdtype(x.dtype, str):
        return (0 - 0.5, len(x) - 0.5)
    try:
        xstep = 0.5 * (x[1] - x[0])
    except IndexError:
        xstep = 0.1
    return (x[0] - xstep, x[-1] + xstep)