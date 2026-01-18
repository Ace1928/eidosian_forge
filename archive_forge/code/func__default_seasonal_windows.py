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
def _default_seasonal_windows(n: int) -> Sequence[int]:
    return tuple((7 + 4 * i for i in range(1, n + 1)))