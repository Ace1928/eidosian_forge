from typing import Optional, Union
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tools.typing import ArrayLike1D
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.tsatools import freq_to_period
def _process_windows(self, windows: Union[int, Sequence[int], None], num_seasons: int) -> Sequence[int]:
    if windows is None:
        windows = self._default_seasonal_windows(num_seasons)
    elif isinstance(windows, int):
        windows = (windows,)
    else:
        pass
    return windows