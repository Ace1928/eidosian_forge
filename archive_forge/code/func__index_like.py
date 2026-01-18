from statsmodels.compat.pandas import (
from abc import ABC, abstractmethod
import datetime as dt
from typing import Optional, Union
from collections.abc import Hashable, Sequence
import numpy as np
import pandas as pd
from scipy.linalg import qr
from statsmodels.iolib.summary import d_or_f
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import freq_to_period
@staticmethod
def _index_like(index: Sequence[Hashable]) -> pd.Index:
    if isinstance(index, pd.Index):
        return index
    try:
        return pd.Index(index)
    except Exception:
        raise TypeError('index must be a pandas Index or index-like')