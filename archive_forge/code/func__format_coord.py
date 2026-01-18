from __future__ import annotations
import functools
from typing import (
import warnings
import numpy as np
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.converter import (
from pandas.tseries.frequencies import (
def _format_coord(freq, t, y) -> str:
    time_period = Period(ordinal=int(t), freq=freq)
    return f't = {time_period}  y = {y:8f}'