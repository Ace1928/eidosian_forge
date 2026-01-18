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
class CalendarFourier(CalendarDeterministicTerm, FourierDeterministicTerm):
    """
    Fourier series deterministic terms based on calendar time

    Parameters
    ----------
    freq : str
        A string convertible to a pandas frequency.
    order : int
        The number of Fourier components to include. Must be <= 2*period.

    See Also
    --------
    DeterministicProcess
    CalendarTimeTrend
    CalendarSeasonality
    Fourier

    Notes
    -----
    Both a sine and a cosine term are included for each i=1, ..., order

    .. math::

       f_{i,s,t} & = \\sin\\left(2 \\pi i \\tau_t \\right)  \\\\
       f_{i,c,t} & = \\cos\\left(2 \\pi i \\tau_t \\right)

    where m is the length of the period and :math:`\\tau_t` is the frequency
    normalized time.  For example, when freq is "D" then an observation with
    a timestamp of 12:00:00 would have :math:`\\tau_t=0.5`.

    Examples
    --------
    Here we simulate irregularly spaced hourly data and construct the calendar
    Fourier terms for the data.

    >>> import numpy as np
    >>> import pandas as pd
    >>> base = pd.Timestamp("2020-1-1")
    >>> gen = np.random.default_rng()
    >>> gaps = np.cumsum(gen.integers(0, 1800, size=1000))
    >>> times = [base + pd.Timedelta(gap, unit="s") for gap in gaps]
    >>> index = pd.DatetimeIndex(pd.to_datetime(times))

    >>> from statsmodels.tsa.deterministic import CalendarFourier
    >>> cal_fourier_gen = CalendarFourier("D", 2)
    >>> cal_fourier_gen.in_sample(index)
    """

    def __init__(self, freq: str, order: int) -> None:
        super().__init__(freq)
        FourierDeterministicTerm.__init__(self, order)
        self._order = required_int_like(order, 'terms')

    @property
    def _columns(self) -> list[str]:
        columns = []
        for i in range(1, self._order + 1):
            for typ in ('sin', 'cos'):
                columns.append(f'{typ}({i},freq={self._freq.freqstr})')
        return columns

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(self, index: Union[Sequence[Hashable], pd.Index]) -> pd.DataFrame:
        index = self._index_like(index)
        index = self._check_index_type(index)
        ratio = self._compute_ratio(index)
        terms = self._get_terms(ratio)
        return pd.DataFrame(terms, index=index, columns=self._columns)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(self, steps: int, index: Union[Sequence[Hashable], pd.Index], forecast_index: Optional[Sequence[Hashable]]=None) -> pd.DataFrame:
        index = self._index_like(index)
        fcast_index = self._extend_index(index, steps, forecast_index)
        self._check_index_type(fcast_index)
        assert isinstance(fcast_index, (pd.DatetimeIndex, pd.PeriodIndex))
        ratio = self._compute_ratio(fcast_index)
        terms = self._get_terms(ratio)
        return pd.DataFrame(terms, index=fcast_index, columns=self._columns)

    @property
    def _eq_attr(self) -> tuple[Hashable, ...]:
        return (self._freq.freqstr, self._order)

    def __str__(self) -> str:
        return f'Fourier(freq={self._freq.freqstr}, order={self._order})'