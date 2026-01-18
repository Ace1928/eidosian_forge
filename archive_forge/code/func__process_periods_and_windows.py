from typing import Optional, Union
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tools.typing import ArrayLike1D
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.tsatools import freq_to_period
def _process_periods_and_windows(self, periods: Union[int, Sequence[int], None], windows: Union[int, Sequence[int], None]) -> tuple[Sequence[int], Sequence[int]]:
    periods = self._process_periods(periods)
    if windows:
        windows = self._process_windows(windows, num_seasons=len(periods))
        periods, windows = self._sort_periods_and_windows(periods, windows)
    else:
        windows = self._process_windows(windows, num_seasons=len(periods))
        periods = sorted(periods)
    if len(periods) != len(windows):
        raise ValueError('Periods and windows must have same length')
    if any((period >= self.nobs / 2 for period in periods)):
        warnings.warn('A period(s) is larger than half the length of time series. Removing these period(s).', UserWarning)
        periods = tuple((period for period in periods if period < self.nobs / 2))
        windows = windows[:len(periods)]
    return (periods, windows)