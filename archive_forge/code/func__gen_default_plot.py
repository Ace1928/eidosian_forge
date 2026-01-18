from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
def _gen_default_plot(f, fig, **kwargs):
    """
    Create plot in a default way.
    """
    yield f(**kwargs)