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
class DeterministicTerm(ABC):
    """Abstract Base Class for all Deterministic Terms"""
    _is_dummy = False

    @property
    def is_dummy(self) -> bool:
        """Flag indicating whether the values produced are dummy variables"""
        return self._is_dummy

    @abstractmethod
    def in_sample(self, index: Sequence[Hashable]) -> pd.DataFrame:
        """
        Produce deterministic trends for in-sample fitting.

        Parameters
        ----------
        index : index_like
            An index-like object. If not an index, it is converted to an
            index.

        Returns
        -------
        DataFrame
            A DataFrame containing the deterministic terms.
        """

    @abstractmethod
    def out_of_sample(self, steps: int, index: Sequence[Hashable], forecast_index: Optional[Sequence[Hashable]]=None) -> pd.DataFrame:
        """
        Produce deterministic trends for out-of-sample forecasts

        Parameters
        ----------
        steps : int
            The number of steps to forecast
        index : index_like
            An index-like object. If not an index, it is converted to an
            index.
        forecast_index : index_like
            An Index or index-like object to use for the forecasts. If
            provided must have steps elements.

        Returns
        -------
        DataFrame
            A DataFrame containing the deterministic terms.
        """

    @abstractmethod
    def __str__(self) -> str:
        """A meaningful string representation of the term"""

    def __hash__(self) -> int:
        name: tuple[Hashable, ...] = (type(self).__name__,)
        return hash(name + self._eq_attr)

    @property
    @abstractmethod
    def _eq_attr(self) -> tuple[Hashable, ...]:
        """tuple of attributes that are used for equality comparison"""

    @staticmethod
    def _index_like(index: Sequence[Hashable]) -> pd.Index:
        if isinstance(index, pd.Index):
            return index
        try:
            return pd.Index(index)
        except Exception:
            raise TypeError('index must be a pandas Index or index-like')

    @staticmethod
    def _extend_index(index: pd.Index, steps: int, forecast_index: Optional[Sequence[Hashable]]=None) -> pd.Index:
        """Extend the forecast index"""
        if forecast_index is not None:
            forecast_index = DeterministicTerm._index_like(forecast_index)
            assert isinstance(forecast_index, pd.Index)
            if forecast_index.shape[0] != steps:
                raise ValueError(f'The number of values in forecast_index ({forecast_index.shape[0]}) must match steps ({steps}).')
            return forecast_index
        if isinstance(index, pd.PeriodIndex):
            return pd.period_range(index[-1] + 1, periods=steps, freq=index.freq)
        elif isinstance(index, pd.DatetimeIndex) and index.freq is not None:
            next_obs = pd.date_range(index[-1], freq=index.freq, periods=2)[1]
            return pd.date_range(next_obs, freq=index.freq, periods=steps)
        elif isinstance(index, pd.RangeIndex):
            assert isinstance(index, pd.RangeIndex)
            try:
                step = index.step
                start = index.stop
            except AttributeError:
                step = index[-1] - index[-2] if len(index) > 1 else 1
                start = index[-1] + step
            stop = start + step * steps
            return pd.RangeIndex(start, stop, step=step)
        elif is_int_index(index) and np.all(np.diff(index) == 1):
            idx_arr = np.arange(index[-1] + 1, index[-1] + steps + 1)
            return pd.Index(idx_arr)
        import warnings
        warnings.warn('Only PeriodIndexes, DatetimeIndexes with a frequency set, RangesIndexes, and Index with a unit increment support extending. The index is set will contain the position relative to the data length.', UserWarning, stacklevel=2)
        nobs = index.shape[0]
        return pd.RangeIndex(nobs + 1, nobs + steps + 1)

    def __repr__(self) -> str:
        return self.__str__() + f' at 0x{id(self):0x}'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            own_attr = self._eq_attr
            oth_attr = other._eq_attr
            if len(own_attr) != len(oth_attr):
                return False
            return all([a == b for a, b in zip(own_attr, oth_attr)])
        else:
            return False