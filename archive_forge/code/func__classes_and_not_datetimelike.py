from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import conversion
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCIndex
from pandas.core.dtypes.inference import (
def _classes_and_not_datetimelike(*klasses) -> Callable:
    """
    Evaluate if the tipo is a subclass of the klasses
    and not a datetimelike.
    """
    return lambda tipo: issubclass(tipo, klasses) and (not issubclass(tipo, (np.datetime64, np.timedelta64)))