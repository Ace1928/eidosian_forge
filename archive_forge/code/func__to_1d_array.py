from typing import Optional, Union
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tools.typing import ArrayLike1D
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.tsatools import freq_to_period
@staticmethod
def _to_1d_array(x):
    y = np.ascontiguousarray(np.squeeze(np.asarray(x)), dtype=np.double)
    if y.ndim != 1:
        raise ValueError('y must be a 1d array')
    return y