from __future__ import annotations
import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, Optional, ClassVar
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import (
from matplotlib.dates import (
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series
from seaborn._core.rules import categorical_order
from seaborn._core.typing import Default, default
from typing import TYPE_CHECKING
def _make_symlog_transforms(c: float=1, base: float=10) -> TransFuncs:
    log, exp = _make_log_transforms(base)

    def symlog(x):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.sign(x) * log(1 + np.abs(np.divide(x, c)))

    def symexp(x):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.sign(x) * c * (exp(np.abs(x)) - 1)
    return (symlog, symexp)