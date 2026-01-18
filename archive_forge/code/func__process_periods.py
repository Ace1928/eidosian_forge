from typing import Optional, Union
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tools.typing import ArrayLike1D
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.tsatools import freq_to_period
def _process_periods(self, periods: Union[int, Sequence[int], None]) -> Sequence[int]:
    if periods is None:
        periods = (self._infer_period(),)
    elif isinstance(periods, int):
        periods = (periods,)
    else:
        pass
    return periods