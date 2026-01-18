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
def _adjust_dummies(self, terms: list[pd.DataFrame]) -> list[pd.DataFrame]:
    has_const: Optional[bool] = None
    for dterm in self._deterministic_terms:
        if isinstance(dterm, (TimeTrend, CalendarTimeTrend)):
            has_const = has_const or dterm.constant
    if has_const is None:
        has_const = False
        for term in terms:
            const_col = (term == term.iloc[0]).all() & (term.iloc[0] != 0)
            has_const = has_const or const_col.any()
    drop_first = has_const
    for i, dterm in enumerate(self._deterministic_terms):
        is_dummy = dterm.is_dummy
        if is_dummy and drop_first:
            terms[i] = terms[i].iloc[:, 1:]
        drop_first = drop_first or is_dummy
    return terms