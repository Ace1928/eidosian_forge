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
def _parse_for_log_params(self, trans: str | TransFuncs | None) -> tuple[float | None, float | None]:
    log_base = symlog_thresh = None
    if isinstance(trans, str):
        m = re.match('^log(\\d*)', trans)
        if m is not None:
            log_base = float(m[1] or 10)
        m = re.match('symlog(\\d*)', trans)
        if m is not None:
            symlog_thresh = float(m[1] or 1)
    return (log_base, symlog_thresh)