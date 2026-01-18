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
class DeterministicProcess:
    """
    Container class for deterministic terms.

    Directly supports constants, time trends, and either seasonal dummies or
    fourier terms for a single cycle. Additional deterministic terms beyond
    the set that can be directly initialized through the constructor can be
    added.

    Parameters
    ----------
    index : {Sequence[Hashable], pd.Index}
        The index of the process. Should usually be the "in-sample" index when
        used in forecasting applications.
    period : {float, int}, default None
        The period of the seasonal or fourier components. Must be an int for
        seasonal dummies. If not provided, freq is read from index if
        available.
    constant : bool, default False
        Whether to include a constant.
    order : int, default 0
        The order of the tim trend to include. For example, 2 will include
        both linear and quadratic terms. 0 exclude time trend terms.
    seasonal : bool = False
        Whether to include seasonal dummies
    fourier : int = 0
        The order of the fourier terms to included.
    additional_terms : Sequence[DeterministicTerm]
        A sequence of additional deterministic terms to include in the process.
    drop : bool, default False
        A flag indicating to check for perfect collinearity and to drop any
        linearly dependent terms.

    See Also
    --------
    TimeTrend
    Seasonality
    Fourier
    CalendarTimeTrend
    CalendarSeasonality
    CalendarFourier

    Notes
    -----
    See the notebook `Deterministic Terms in Time Series Models
    <../examples/notebooks/generated/deterministics.html>`__ for an overview.

    Examples
    --------
    >>> from statsmodels.tsa.deterministic import DeterministicProcess
    >>> from pandas import date_range
    >>> index = date_range("2000-1-1", freq="M", periods=240)

    First a determinstic process with a constant and quadratic time trend.

    >>> dp = DeterministicProcess(index, constant=True, order=2)
    >>> dp.in_sample().head(3)
                const  trend  trend_squared
    2000-01-31    1.0    1.0            1.0
    2000-02-29    1.0    2.0            4.0
    2000-03-31    1.0    3.0            9.0

    Seasonal dummies are included by setting seasonal to True.

    >>> dp = DeterministicProcess(index, constant=True, seasonal=True)
    >>> dp.in_sample().iloc[:3,:5]
                const  s(2,12)  s(3,12)  s(4,12)  s(5,12)
    2000-01-31    1.0      0.0      0.0      0.0      0.0
    2000-02-29    1.0      1.0      0.0      0.0      0.0
    2000-03-31    1.0      0.0      1.0      0.0      0.0

    Fourier components can be used to alternatively capture seasonal patterns,

    >>> dp = DeterministicProcess(index, constant=True, fourier=2)
    >>> dp.in_sample().head(3)
                const  sin(1,12)  cos(1,12)  sin(2,12)  cos(2,12)
    2000-01-31    1.0   0.000000   1.000000   0.000000        1.0
    2000-02-29    1.0   0.500000   0.866025   0.866025        0.5
    2000-03-31    1.0   0.866025   0.500000   0.866025       -0.5

    Multiple Seasonalities can be captured using additional terms.

    >>> from statsmodels.tsa.deterministic import Fourier
    >>> index = date_range("2000-1-1", freq="D", periods=5000)
    >>> fourier = Fourier(period=365.25, order=1)
    >>> dp = DeterministicProcess(index, period=3, constant=True,
    ...                           seasonal=True, additional_terms=[fourier])
    >>> dp.in_sample().head(3)
                const  s(2,3)  s(3,3)  sin(1,365.25)  cos(1,365.25)
    2000-01-01    1.0     0.0     0.0       0.000000       1.000000
    2000-01-02    1.0     1.0     0.0       0.017202       0.999852
    2000-01-03    1.0     0.0     1.0       0.034398       0.999408
    """

    def __init__(self, index: Union[Sequence[Hashable], pd.Index], *, period: Optional[Union[float, int]]=None, constant: bool=False, order: int=0, seasonal: bool=False, fourier: int=0, additional_terms: Sequence[DeterministicTerm]=(), drop: bool=False):
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        self._index = index
        self._deterministic_terms: list[DeterministicTerm] = []
        self._extendable = False
        self._index_freq = None
        self._validate_index()
        period = float_like(period, 'period', optional=True)
        self._constant = constant = bool_like(constant, 'constant')
        self._order = required_int_like(order, 'order')
        self._seasonal = seasonal = bool_like(seasonal, 'seasonal')
        self._fourier = required_int_like(fourier, 'fourier')
        additional_terms = tuple(additional_terms)
        self._cached_in_sample = None
        self._drop = bool_like(drop, 'drop')
        self._additional_terms = additional_terms
        if constant or order:
            self._deterministic_terms.append(TimeTrend(constant, order))
        if seasonal and fourier:
            raise ValueError('seasonal and fourier can be initialized through the constructor since these will be necessarily perfectly collinear. Instead, you can pass additional components using the additional_terms input.')
        if (seasonal or fourier) and period is None:
            if period is None:
                self._period = period = freq_to_period(self._index_freq)
        if seasonal:
            period = required_int_like(period, 'period')
            self._deterministic_terms.append(Seasonality(period))
        elif fourier:
            period = float_like(period, 'period')
            assert period is not None
            self._deterministic_terms.append(Fourier(period, order=fourier))
        for term in additional_terms:
            if not isinstance(term, DeterministicTerm):
                raise TypeError('All additional terms must be instances of subsclasses of DeterministicTerm')
            if term not in self._deterministic_terms:
                self._deterministic_terms.append(term)
            else:
                raise ValueError('One or more terms in additional_terms has been added through the parameters of the constructor. Terms must be unique.')
        self._period = period
        self._retain_cols: Optional[list[Hashable]] = None

    @property
    def index(self) -> pd.Index:
        """The index of the process"""
        return self._index

    @property
    def terms(self) -> list[DeterministicTerm]:
        """The deterministic terms included in the process"""
        return self._deterministic_terms

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

    def _remove_zeros_ones(self, terms: pd.DataFrame) -> pd.DataFrame:
        all_zero = np.all(terms == 0, axis=0)
        if np.any(all_zero):
            terms = terms.loc[:, ~all_zero]
        is_constant = terms.max(axis=0) == terms.min(axis=0)
        if np.sum(is_constant) > 1:
            surplus_consts = is_constant & is_constant.duplicated()
            terms = terms.loc[:, ~surplus_consts]
        return terms

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(self) -> pd.DataFrame:
        if self._cached_in_sample is not None:
            return self._cached_in_sample
        index = self._index
        if not self._deterministic_terms:
            return pd.DataFrame(np.empty((index.shape[0], 0)), index=index)
        raw_terms = []
        for term in self._deterministic_terms:
            raw_terms.append(term.in_sample(index))
        raw_terms = self._adjust_dummies(raw_terms)
        terms: pd.DataFrame = pd.concat(raw_terms, axis=1)
        terms = self._remove_zeros_ones(terms)
        if self._drop:
            terms_arr = to_numpy(terms)
            res = qr(terms_arr, mode='r', pivoting=True)
            r = res[0]
            p = res[-1]
            abs_diag = np.abs(np.diag(r))
            tol = abs_diag[0] * terms_arr.shape[1] * np.finfo(float).eps
            rank = int(np.sum(abs_diag > tol))
            rpx = r.T @ terms_arr
            keep = [0]
            last_rank = 1
            for i in range(1, terms_arr.shape[1]):
                curr_rank = np.linalg.matrix_rank(rpx[:i + 1, :i + 1])
                if curr_rank > last_rank:
                    keep.append(i)
                    last_rank = curr_rank
                if curr_rank == rank:
                    break
            if len(keep) == rank:
                terms = terms.iloc[:, keep]
            else:
                terms = terms.iloc[:, np.sort(p[:rank])]
        self._retain_cols = terms.columns
        self._cached_in_sample = terms
        return terms

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(self, steps: int, forecast_index: Optional[Union[Sequence[Hashable], pd.Index]]=None) -> pd.DataFrame:
        steps = required_int_like(steps, 'steps')
        if self._drop and self._retain_cols is None:
            self.in_sample()
        index = self._index
        if not self._deterministic_terms:
            return pd.DataFrame(np.empty((index.shape[0], 0)), index=index)
        raw_terms = []
        for term in self._deterministic_terms:
            raw_terms.append(term.out_of_sample(steps, index, forecast_index))
        terms: pd.DataFrame = pd.concat(raw_terms, axis=1)
        assert self._retain_cols is not None
        if terms.shape[1] != len(self._retain_cols):
            terms = terms[self._retain_cols]
        return terms

    def _extend_time_index(self, stop: pd.Timestamp) -> Union[pd.DatetimeIndex, pd.PeriodIndex]:
        index = self._index
        if isinstance(index, pd.PeriodIndex):
            return pd.period_range(index[0], end=stop, freq=index.freq)
        return pd.date_range(start=index[0], end=stop, freq=self._index_freq)

    def _range_from_range_index(self, start: int, stop: int) -> pd.DataFrame:
        index = self._index
        is_int64_index = is_int_index(index)
        assert isinstance(index, pd.RangeIndex) or is_int64_index
        if start < index[0]:
            raise ValueError(START_BEFORE_INDEX_ERR)
        if isinstance(index, pd.RangeIndex):
            idx_step = index.step
        else:
            idx_step = np.diff(index).max() if len(index) > 1 else 1
        if idx_step != 1 and (start - index[0]) % idx_step != 0:
            raise ValueError(f'The step of the index is not 1 (actual step={idx_step}). start must be in the sequence that would have been generated by the index.')
        if is_int64_index:
            new_idx = pd.Index(np.arange(start, stop))
        else:
            new_idx = pd.RangeIndex(start, stop, step=idx_step)
        if new_idx[-1] <= self._index[-1]:
            in_sample = self.in_sample()
            in_sample = in_sample.loc[new_idx]
            return in_sample
        elif new_idx[0] > self._index[-1]:
            next_value = index[-1] + idx_step
            if new_idx[0] != next_value:
                tmp = pd.RangeIndex(next_value, stop, step=idx_step)
                oos = self.out_of_sample(tmp.shape[0], forecast_index=tmp)
                return oos.loc[new_idx]
            return self.out_of_sample(new_idx.shape[0], forecast_index=new_idx)
        in_sample_loc = new_idx <= self._index[-1]
        in_sample_idx = new_idx[in_sample_loc]
        out_of_sample_idx = new_idx[~in_sample_loc]
        in_sample_exog = self.in_sample().loc[in_sample_idx]
        oos_exog = self.out_of_sample(steps=out_of_sample_idx.shape[0], forecast_index=out_of_sample_idx)
        return pd.concat([in_sample_exog, oos_exog], axis=0)

    def _range_from_time_index(self, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
        index = self._index
        if isinstance(self._index, pd.PeriodIndex):
            if isinstance(start, pd.Timestamp):
                start = start.to_period(freq=self._index_freq)
            if isinstance(stop, pd.Timestamp):
                stop = stop.to_period(freq=self._index_freq)
        if start < index[0]:
            raise ValueError(START_BEFORE_INDEX_ERR)
        if stop <= self._index[-1]:
            return self.in_sample().loc[start:stop]
        new_idx = self._extend_time_index(stop)
        oos_idx = new_idx[new_idx > index[-1]]
        oos = self.out_of_sample(oos_idx.shape[0], oos_idx)
        if start >= oos_idx[0]:
            return oos.loc[start:stop]
        both = pd.concat([self.in_sample(), oos], axis=0)
        return both.loc[start:stop]

    def _int_to_timestamp(self, value: int, name: str) -> pd.Timestamp:
        if value < 0:
            raise ValueError(f'{name} must be non-negative.')
        if value < self._index.shape[0]:
            return self._index[value]
        add_periods = value - (self._index.shape[0] - 1) + 1
        index = self._index
        if isinstance(self._index, pd.PeriodIndex):
            pr = pd.period_range(index[-1], freq=self._index_freq, periods=add_periods)
            return pr[-1].to_timestamp()
        dr = pd.date_range(index[-1], freq=self._index_freq, periods=add_periods)
        return dr[-1]

    def range(self, start: Union[IntLike, DateLike, str], stop: Union[IntLike, DateLike, str]) -> pd.DataFrame:
        """
        Deterministic terms spanning a range of observations

        Parameters
        ----------
        start : {int, str, dt.datetime, pd.Timestamp, np.datetime64}
            The first observation.
        stop : {int, str, dt.datetime, pd.Timestamp, np.datetime64}
            The final observation. Inclusive to match most prediction
            function in statsmodels.

        Returns
        -------
        DataFrame
            A data frame of deterministic terms
        """
        if not self._extendable:
            raise TypeError('The index in the deterministic process does not support extension. Only PeriodIndex, DatetimeIndex with a frequency, RangeIndex, and integral Indexes that start at 0 and have only unit differences can be extended when producing out-of-sample forecasts.\n')
        if type(self._index) in (pd.RangeIndex,) or is_int_index(self._index):
            start = required_int_like(start, 'start')
            stop = required_int_like(stop, 'stop')
            stop += 1
            return self._range_from_range_index(start, stop)
        if isinstance(start, (int, np.integer)):
            start = self._int_to_timestamp(start, 'start')
        else:
            start = pd.Timestamp(start)
        if isinstance(stop, (int, np.integer)):
            stop = self._int_to_timestamp(stop, 'stop')
        else:
            stop = pd.Timestamp(stop)
        return self._range_from_time_index(start, stop)

    def _validate_index(self) -> None:
        if isinstance(self._index, pd.PeriodIndex):
            self._index_freq = self._index.freq
            self._extendable = True
        elif isinstance(self._index, pd.DatetimeIndex):
            self._index_freq = self._index.freq or self._index.inferred_freq
            self._extendable = self._index_freq is not None
        elif isinstance(self._index, pd.RangeIndex):
            self._extendable = True
        elif is_int_index(self._index):
            self._extendable = self._index[0] == 0 and np.all(np.diff(self._index) == 1)

    def apply(self, index):
        """
        Create an identical determinstic process with a different index

        Parameters
        ----------
        index : index_like
            An index-like object. If not an index, it is converted to an
            index.

        Returns
        -------
        DeterministicProcess
            The deterministic process applied to a different index
        """
        return DeterministicProcess(index, period=self._period, constant=self._constant, order=self._order, seasonal=self._seasonal, fourier=self._fourier, additional_terms=self._additional_terms, drop=self._drop)