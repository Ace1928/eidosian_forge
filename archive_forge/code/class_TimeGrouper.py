from __future__ import annotations
import copy
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import NDFrameT
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
import pandas.core.algorithms as algos
from pandas.core.apply import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.generic import (
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import (
from pandas.tseries.frequencies import (
from pandas.tseries.offsets import (
class TimeGrouper(Grouper):
    """
    Custom groupby class for time-interval grouping.

    Parameters
    ----------
    freq : pandas date offset or offset alias for identifying bin edges
    closed : closed end of interval; 'left' or 'right'
    label : interval boundary to use for labeling; 'left' or 'right'
    convention : {'start', 'end', 'e', 's'}
        If axis is PeriodIndex
    """
    _attributes = Grouper._attributes + ('closed', 'label', 'how', 'kind', 'convention', 'origin', 'offset')
    origin: TimeGrouperOrigin

    def __init__(self, obj: Grouper | None=None, freq: Frequency='Min', key: str | None=None, closed: Literal['left', 'right'] | None=None, label: Literal['left', 'right'] | None=None, how: str='mean', axis: Axis=0, fill_method=None, limit: int | None=None, kind: str | None=None, convention: Literal['start', 'end', 'e', 's'] | None=None, origin: Literal['epoch', 'start', 'start_day', 'end', 'end_day'] | TimestampConvertibleTypes='start_day', offset: TimedeltaConvertibleTypes | None=None, group_keys: bool=False, **kwargs) -> None:
        if label not in {None, 'left', 'right'}:
            raise ValueError(f'Unsupported value {label} for `label`')
        if closed not in {None, 'left', 'right'}:
            raise ValueError(f'Unsupported value {closed} for `closed`')
        if convention not in {None, 'start', 'end', 'e', 's'}:
            raise ValueError(f'Unsupported value {convention} for `convention`')
        if key is None and obj is not None and isinstance(obj.index, PeriodIndex) or (key is not None and obj is not None and (getattr(obj[key], 'dtype', None) == 'period')):
            freq = to_offset(freq, is_period=True)
        else:
            freq = to_offset(freq)
        end_types = {'ME', 'YE', 'QE', 'BME', 'BYE', 'BQE', 'W'}
        rule = freq.rule_code
        if rule in end_types or ('-' in rule and rule[:rule.find('-')] in end_types):
            if closed is None:
                closed = 'right'
            if label is None:
                label = 'right'
        elif origin in ['end', 'end_day']:
            if closed is None:
                closed = 'right'
            if label is None:
                label = 'right'
        else:
            if closed is None:
                closed = 'left'
            if label is None:
                label = 'left'
        self.closed = closed
        self.label = label
        self.kind = kind
        self.convention = convention if convention is not None else 'e'
        self.how = how
        self.fill_method = fill_method
        self.limit = limit
        self.group_keys = group_keys
        self._arrow_dtype: ArrowDtype | None = None
        if origin in ('epoch', 'start', 'start_day', 'end', 'end_day'):
            self.origin = origin
        else:
            try:
                self.origin = Timestamp(origin)
            except (ValueError, TypeError) as err:
                raise ValueError(f"'origin' should be equal to 'epoch', 'start', 'start_day', 'end', 'end_day' or should be a Timestamp convertible type. Got '{origin}' instead.") from err
        try:
            self.offset = Timedelta(offset) if offset is not None else None
        except (ValueError, TypeError) as err:
            raise ValueError(f"'offset' should be a Timedelta convertible type. Got '{offset}' instead.") from err
        kwargs['sort'] = True
        super().__init__(freq=freq, key=key, axis=axis, **kwargs)

    def _get_resampler(self, obj: NDFrame, kind=None) -> Resampler:
        """
        Return my resampler or raise if we have an invalid axis.

        Parameters
        ----------
        obj : Series or DataFrame
        kind : string, optional
            'period','timestamp','timedelta' are valid

        Returns
        -------
        Resampler

        Raises
        ------
        TypeError if incompatible axis

        """
        _, ax, _ = self._set_grouper(obj, gpr_index=None)
        if isinstance(ax, DatetimeIndex):
            return DatetimeIndexResampler(obj, timegrouper=self, kind=kind, axis=self.axis, group_keys=self.group_keys, gpr_index=ax)
        elif isinstance(ax, PeriodIndex) or kind == 'period':
            if isinstance(ax, PeriodIndex):
                warnings.warn('Resampling with a PeriodIndex is deprecated. Cast index to DatetimeIndex before resampling instead.', FutureWarning, stacklevel=find_stack_level())
            else:
                warnings.warn("Resampling with kind='period' is deprecated.  Use datetime paths instead.", FutureWarning, stacklevel=find_stack_level())
            return PeriodIndexResampler(obj, timegrouper=self, kind=kind, axis=self.axis, group_keys=self.group_keys, gpr_index=ax)
        elif isinstance(ax, TimedeltaIndex):
            return TimedeltaIndexResampler(obj, timegrouper=self, axis=self.axis, group_keys=self.group_keys, gpr_index=ax)
        raise TypeError(f"Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of '{type(ax).__name__}'")

    def _get_grouper(self, obj: NDFrameT, validate: bool=True) -> tuple[BinGrouper, NDFrameT]:
        r = self._get_resampler(obj)
        return (r._grouper, cast(NDFrameT, r.obj))

    def _get_time_bins(self, ax: DatetimeIndex):
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(f'axis must be a DatetimeIndex, but got an instance of {type(ax).__name__}')
        if len(ax) == 0:
            binner = labels = DatetimeIndex(data=[], freq=self.freq, name=ax.name, dtype=ax.dtype)
            return (binner, [], labels)
        first, last = _get_timestamp_range_edges(ax.min(), ax.max(), self.freq, unit=ax.unit, closed=self.closed, origin=self.origin, offset=self.offset)
        binner = labels = date_range(freq=self.freq, start=first, end=last, tz=ax.tz, name=ax.name, ambiguous=True, nonexistent='shift_forward', unit=ax.unit)
        ax_values = ax.asi8
        binner, bin_edges = self._adjust_bin_edges(binner, ax_values)
        bins = lib.generate_bins_dt64(ax_values, bin_edges, self.closed, hasnans=ax.hasnans)
        if self.closed == 'right':
            labels = binner
            if self.label == 'right':
                labels = labels[1:]
        elif self.label == 'right':
            labels = labels[1:]
        if ax.hasnans:
            binner = binner.insert(0, NaT)
            labels = labels.insert(0, NaT)
        if len(bins) < len(labels):
            labels = labels[:len(bins)]
        return (binner, bins, labels)

    def _adjust_bin_edges(self, binner: DatetimeIndex, ax_values: npt.NDArray[np.int64]) -> tuple[DatetimeIndex, npt.NDArray[np.int64]]:
        if self.freq.name in ('BME', 'ME', 'W') or self.freq.name.split('-')[0] in ('BQE', 'BYE', 'QE', 'YE', 'W'):
            if self.closed == 'right':
                edges_dti = binner.tz_localize(None)
                edges_dti = edges_dti + Timedelta(days=1, unit=edges_dti.unit).as_unit(edges_dti.unit) - Timedelta(1, unit=edges_dti.unit).as_unit(edges_dti.unit)
                bin_edges = edges_dti.tz_localize(binner.tz).asi8
            else:
                bin_edges = binner.asi8
            if bin_edges[-2] > ax_values.max():
                bin_edges = bin_edges[:-1]
                binner = binner[:-1]
        else:
            bin_edges = binner.asi8
        return (binner, bin_edges)

    def _get_time_delta_bins(self, ax: TimedeltaIndex):
        if not isinstance(ax, TimedeltaIndex):
            raise TypeError(f'axis must be a TimedeltaIndex, but got an instance of {type(ax).__name__}')
        if not isinstance(self.freq, Tick):
            raise ValueError(f"Resampling on a TimedeltaIndex requires fixed-duration `freq`, e.g. '24h' or '3D', not {self.freq}")
        if not len(ax):
            binner = labels = TimedeltaIndex(data=[], freq=self.freq, name=ax.name)
            return (binner, [], labels)
        start, end = (ax.min(), ax.max())
        if self.closed == 'right':
            end += self.freq
        labels = binner = timedelta_range(start=start, end=end, freq=self.freq, name=ax.name)
        end_stamps = labels
        if self.closed == 'left':
            end_stamps += self.freq
        bins = ax.searchsorted(end_stamps, side=self.closed)
        if self.offset:
            labels += self.offset
        return (binner, bins, labels)

    def _get_time_period_bins(self, ax: DatetimeIndex):
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(f'axis must be a DatetimeIndex, but got an instance of {type(ax).__name__}')
        freq = self.freq
        if len(ax) == 0:
            binner = labels = PeriodIndex(data=[], freq=freq, name=ax.name, dtype=ax.dtype)
            return (binner, [], labels)
        labels = binner = period_range(start=ax[0], end=ax[-1], freq=freq, name=ax.name)
        end_stamps = (labels + freq).asfreq(freq, 's').to_timestamp()
        if ax.tz:
            end_stamps = end_stamps.tz_localize(ax.tz)
        bins = ax.searchsorted(end_stamps, side='left')
        return (binner, bins, labels)

    def _get_period_bins(self, ax: PeriodIndex):
        if not isinstance(ax, PeriodIndex):
            raise TypeError(f'axis must be a PeriodIndex, but got an instance of {type(ax).__name__}')
        memb = ax.asfreq(self.freq, how=self.convention)
        nat_count = 0
        if memb.hasnans:
            nat_count = np.sum(memb._isnan)
            memb = memb[~memb._isnan]
        if not len(memb):
            bins = np.array([], dtype=np.int64)
            binner = labels = PeriodIndex(data=[], freq=self.freq, name=ax.name)
            if len(ax) > 0:
                binner, bins, labels = _insert_nat_bin(binner, bins, labels, len(ax))
            return (binner, bins, labels)
        freq_mult = self.freq.n
        start = ax.min().asfreq(self.freq, how=self.convention)
        end = ax.max().asfreq(self.freq, how='end')
        bin_shift = 0
        if isinstance(self.freq, Tick):
            p_start, end = _get_period_range_edges(start, end, self.freq, closed=self.closed, origin=self.origin, offset=self.offset)
            start_offset = Period(start, self.freq) - Period(p_start, self.freq)
            bin_shift = start_offset.n % freq_mult
            start = p_start
        labels = binner = period_range(start=start, end=end, freq=self.freq, name=ax.name)
        i8 = memb.asi8
        expected_bins_count = len(binner) * freq_mult
        i8_extend = expected_bins_count - (i8[-1] - i8[0])
        rng = np.arange(i8[0], i8[-1] + i8_extend, freq_mult)
        rng += freq_mult
        rng -= bin_shift
        prng = type(memb._data)(rng, dtype=memb.dtype)
        bins = memb.searchsorted(prng, side='left')
        if nat_count > 0:
            binner, bins, labels = _insert_nat_bin(binner, bins, labels, nat_count)
        return (binner, bins, labels)

    def _set_grouper(self, obj: NDFrameT, sort: bool=False, *, gpr_index: Index | None=None) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
        obj, ax, indexer = super()._set_grouper(obj, sort, gpr_index=gpr_index)
        if isinstance(ax.dtype, ArrowDtype) and ax.dtype.kind in 'Mm':
            self._arrow_dtype = ax.dtype
            ax = Index(cast(ArrowExtensionArray, ax.array)._maybe_convert_datelike_array())
        return (obj, ax, indexer)