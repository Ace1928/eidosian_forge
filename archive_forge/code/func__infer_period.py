from typing import Optional, Union
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tools.typing import ArrayLike1D
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.tsatools import freq_to_period
def _infer_period(self) -> int:
    freq = None
    if isinstance(self.endog, (pd.Series, pd.DataFrame)):
        freq = getattr(self.endog.index, 'inferred_freq', None)
    if freq is None:
        raise ValueError('Unable to determine period from endog')
    period = freq_to_period(freq)
    return period