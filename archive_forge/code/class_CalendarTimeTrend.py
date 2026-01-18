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
class CalendarTimeTrend(CalendarDeterministicTerm, TimeTrendDeterministicTerm):
    """
    Constant and time trend determinstic terms based on calendar time

    Parameters
    ----------
    freq : str
        A string convertible to a pandas frequency.
    constant : bool
        Flag indicating whether a constant should be included.
    order : int
        A non-negative int containing the powers to include (1, 2, ..., order).
    base_period : {str, pd.Timestamp}, default None
        The base period to use when computing the time stamps. This value is
        treated as 1 and so all other time indices are defined as the number
        of periods since or before this time stamp. If not provided, defaults
        to pandas base period for a PeriodIndex.

    See Also
    --------
    DeterministicProcess
    CalendarFourier
    CalendarSeasonality
    TimeTrend

    Notes
    -----
    The time stamp, :math:`\\tau_t`, is the number of periods that have elapsed
    since the base_period. :math:`\\tau_t` may be fractional.

    Examples
    --------
    Here we simulate irregularly spaced hourly data and construct the calendar
    time trend terms for the data.

    >>> import numpy as np
    >>> import pandas as pd
    >>> base = pd.Timestamp("2020-1-1")
    >>> gen = np.random.default_rng()
    >>> gaps = np.cumsum(gen.integers(0, 1800, size=1000))
    >>> times = [base + pd.Timedelta(gap, unit="s") for gap in gaps]
    >>> index = pd.DatetimeIndex(pd.to_datetime(times))

    >>> from statsmodels.tsa.deterministic import CalendarTimeTrend
    >>> cal_trend_gen = CalendarTimeTrend("D", True, order=1)
    >>> cal_trend_gen.in_sample(index)

    Next, we normalize using the first time stamp

    >>> cal_trend_gen = CalendarTimeTrend("D", True, order=1,
    ...                                   base_period=index[0])
    >>> cal_trend_gen.in_sample(index)
    """

    def __init__(self, freq: str, constant: bool=True, order: int=0, *, base_period: Optional[Union[str, DateLike]]=None) -> None:
        super().__init__(freq)
        TimeTrendDeterministicTerm.__init__(self, constant=constant, order=order)
        self._ref_i8 = 0
        if base_period is not None:
            pr = pd.period_range(base_period, periods=1, freq=self._freq)
            self._ref_i8 = pr.asi8[0]
        self._base_period = None if base_period is None else str(base_period)

    @property
    def base_period(self) -> Optional[str]:
        """The base period"""
        return self._base_period

    @classmethod
    def from_string(cls, freq: str, trend: str, base_period: Optional[Union[str, DateLike]]=None) -> 'CalendarTimeTrend':
        """
        Create a TimeTrend from a string description.

        Provided for compatibility with common string names.

        Parameters
        ----------
        freq : str
            A string convertible to a pandas frequency.
        trend : {"n", "c", "t", "ct", "ctt"}
            The string representation of the time trend. The terms are:

            * "n": No trend terms
            * "c": A constant only
            * "t": Linear time trend only
            * "ct": A constant and a time trend
            * "ctt": A constant, a time trend and a quadratic time trend
        base_period : {str, pd.Timestamp}, default None
            The base period to use when computing the time stamps. This value
            is treated as 1 and so all other time indices are defined as the
            number of periods since or before this time stamp. If not
            provided, defaults to pandas base period for a PeriodIndex.

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
        return cls(freq, constant, order, base_period=base_period)

    def _terms(self, index: Union[pd.DatetimeIndex, pd.PeriodIndex], ratio: np.ndarray) -> pd.DataFrame:
        if isinstance(index, pd.DatetimeIndex):
            index = index.to_period(self._freq)
        index_i8 = index.asi8
        index_i8 = index_i8 - self._ref_i8 + 1
        time = index_i8.astype(np.double) + ratio
        time = time[:, None]
        terms = self._get_terms(time)
        return pd.DataFrame(terms, columns=self._columns, index=index)

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(self, index: Union[Sequence[Hashable], pd.Index]) -> pd.DataFrame:
        index = self._index_like(index)
        index = self._check_index_type(index)
        ratio = self._compute_ratio(index)
        return self._terms(index, ratio)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(self, steps: int, index: Union[Sequence[Hashable], pd.Index], forecast_index: Optional[Sequence[Hashable]]=None) -> pd.DataFrame:
        index = self._index_like(index)
        fcast_index = self._extend_index(index, steps, forecast_index)
        self._check_index_type(fcast_index)
        assert isinstance(fcast_index, (pd.PeriodIndex, pd.DatetimeIndex))
        ratio = self._compute_ratio(fcast_index)
        return self._terms(fcast_index, ratio)

    @property
    def _eq_attr(self) -> tuple[Hashable, ...]:
        attr: tuple[Hashable, ...] = (self._constant, self._order, self._freq.freqstr)
        if self._base_period is not None:
            attr += (self._base_period,)
        return attr

    def __str__(self) -> str:
        value = TimeTrendDeterministicTerm.__str__(self)
        value = 'Calendar' + value[:-1] + f', freq={self._freq.freqstr})'
        if self._base_period is not None:
            value = value[:-1] + f'base_period={self._base_period})'
        return value