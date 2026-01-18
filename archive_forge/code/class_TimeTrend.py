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
class TimeTrend(TimeTrendDeterministicTerm):
    """
    Constant and time trend determinstic terms

    Parameters
    ----------
    constant : bool
        Flag indicating whether a constant should be included.
    order : int
        A non-negative int containing the powers to include (1, 2, ..., order).

    See Also
    --------
    DeterministicProcess
    Seasonality
    Fourier
    CalendarTimeTrend

    Examples
    --------
    >>> from statsmodels.datasets import sunspots
    >>> from statsmodels.tsa.deterministic import TimeTrend
    >>> data = sunspots.load_pandas().data
    >>> trend_gen = TimeTrend(True, 3)
    >>> trend_gen.in_sample(data.index)
    """

    def __init__(self, constant: bool=True, order: int=0) -> None:
        super().__init__(constant, order)

    @classmethod
    def from_string(cls, trend: str) -> 'TimeTrend':
        """
        Create a TimeTrend from a string description.

        Provided for compatibility with common string names.

        Parameters
        ----------
        trend : {"n", "c", "t", "ct", "ctt"}
            The string representation of the time trend. The terms are:

            * "n": No trend terms
            * "c": A constant only
            * "t": Linear time trend only
            * "ct": A constant and a time trend
            * "ctt": A constant, a time trend and a quadratic time trend

        Returns
        -------
        TimeTrend
            The TimeTrend instance.
        """
        constant = trend.startswith('c')
        order = 0
        if 'tt' in trend:
            order = 2
        elif 't' in trend:
            order = 1
        return cls(constant=constant, order=order)

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(self, index: Union[Sequence[Hashable], pd.Index]) -> pd.DataFrame:
        index = self._index_like(index)
        nobs = index.shape[0]
        locs = np.arange(1, nobs + 1, dtype=np.double)[:, None]
        terms = self._get_terms(locs)
        return pd.DataFrame(terms, columns=self._columns, index=index)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(self, steps: int, index: Union[Sequence[Hashable], pd.Index], forecast_index: Optional[Sequence[Hashable]]=None) -> pd.DataFrame:
        index = self._index_like(index)
        nobs = index.shape[0]
        fcast_index = self._extend_index(index, steps, forecast_index)
        locs = np.arange(nobs + 1, nobs + steps + 1, dtype=np.double)[:, None]
        terms = self._get_terms(locs)
        return pd.DataFrame(terms, columns=self._columns, index=fcast_index)

    @property
    def _eq_attr(self) -> tuple[Hashable, ...]:
        return (self._constant, self._order)