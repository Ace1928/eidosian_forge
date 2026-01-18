from __future__ import annotations
import itertools
import types
import typing
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from .._utils import cross_join, match
from ..exceptions import PlotnineError
from ..scales.scales import Scales
from .strips import Strips
def _inf_to_none(t: tuple[float, float]) -> tuple[float | None, float | None]:
    """
            Replace infinities with None
            """
    a = t[0] if np.isfinite(t[0]) else None
    b = t[1] if np.isfinite(t[1]) else None
    return (a, b)