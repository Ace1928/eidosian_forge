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
def _is_sup(f1: str, f2: str) -> bool:
    return f1.startswith('W') and is_superperiod('D', f2) or (f2.startswith('W') and is_superperiod(f1, 'D'))