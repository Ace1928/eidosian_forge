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
def _get_period_alias(freq: timedelta | BaseOffset | str) -> str | None:
    if isinstance(freq, BaseOffset):
        freqstr = freq.name
    else:
        freqstr = to_offset(freq, is_period=True).rule_code
    return get_period_alias(freqstr)