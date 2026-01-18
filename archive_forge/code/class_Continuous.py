from __future__ import annotations
import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, Optional, ClassVar
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import (
from matplotlib.dates import (
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series
from seaborn._core.rules import categorical_order
from seaborn._core.typing import Default, default
from typing import TYPE_CHECKING
@dataclass
class Continuous(ContinuousBase):
    """
    A numeric scale supporting norms and functional transforms.
    """
    values: tuple | str | None = None
    trans: str | TransFuncs | None = None
    _priority: ClassVar[int] = 1

    def tick(self, locator: Locator | None=None, *, at: Sequence[float] | None=None, upto: int | None=None, count: int | None=None, every: float | None=None, between: tuple[float, float] | None=None, minor: int | None=None) -> Continuous:
        """
        Configure the selection of ticks for the scale's axis or legend.

        Parameters
        ----------
        locator : :class:`matplotlib.ticker.Locator` subclass
            Pre-configured matplotlib locator; other parameters will not be used.
        at : sequence of floats
            Place ticks at these specific locations (in data units).
        upto : int
            Choose "nice" locations for ticks, but do not exceed this number.
        count : int
            Choose exactly this number of ticks, bounded by `between` or axis limits.
        every : float
            Choose locations at this interval of separation (in data units).
        between : pair of floats
            Bound upper / lower ticks when using `every` or `count`.
        minor : int
            Number of unlabeled ticks to draw between labeled "major" ticks.

        Returns
        -------
        scale
            Copy of self with new tick configuration.

        """
        if locator is not None and (not isinstance(locator, Locator)):
            raise TypeError(f'Tick locator must be an instance of {Locator!r}, not {type(locator)!r}.')
        log_base, symlog_thresh = self._parse_for_log_params(self.trans)
        if log_base or symlog_thresh:
            if count is not None and between is None:
                raise RuntimeError('`count` requires `between` with log transform.')
            if every is not None:
                raise RuntimeError('`every` not supported with log transform.')
        new = copy(self)
        new._tick_params = {'locator': locator, 'at': at, 'upto': upto, 'count': count, 'every': every, 'between': between, 'minor': minor}
        return new

    def label(self, formatter: Formatter | None=None, *, like: str | Callable | None=None, base: int | None | Default=default, unit: str | None=None) -> Continuous:
        """
        Configure the appearance of tick labels for the scale's axis or legend.

        Parameters
        ----------
        formatter : :class:`matplotlib.ticker.Formatter` subclass
            Pre-configured formatter to use; other parameters will be ignored.
        like : str or callable
            Either a format pattern (e.g., `".2f"`), a format string with fields named
            `x` and/or `pos` (e.g., `"${x:.2f}"`), or a callable with a signature like
            `f(x: float, pos: int) -> str`. In the latter variants, `x` is passed as the
            tick value and `pos` is passed as the tick index.
        base : number
            Use log formatter (with scientific notation) having this value as the base.
            Set to `None` to override the default formatter with a log transform.
        unit : str or (str, str) tuple
            Use  SI prefixes with these units (e.g., with `unit="g"`, a tick value
            of 5000 will appear as `5 kg`). When a tuple, the first element gives the
            separator between the number and unit.

        Returns
        -------
        scale
            Copy of self with new label configuration.

        """
        if formatter is not None and (not isinstance(formatter, Formatter)):
            raise TypeError(f'Label formatter must be an instance of {Formatter!r}, not {type(formatter)!r}')
        if like is not None and (not (isinstance(like, str) or callable(like))):
            msg = f'`like` must be a string or callable, not {type(like).__name__}.'
            raise TypeError(msg)
        new = copy(self)
        new._label_params = {'formatter': formatter, 'like': like, 'base': base, 'unit': unit}
        return new

    def _parse_for_log_params(self, trans: str | TransFuncs | None) -> tuple[float | None, float | None]:
        log_base = symlog_thresh = None
        if isinstance(trans, str):
            m = re.match('^log(\\d*)', trans)
            if m is not None:
                log_base = float(m[1] or 10)
            m = re.match('symlog(\\d*)', trans)
            if m is not None:
                symlog_thresh = float(m[1] or 1)
        return (log_base, symlog_thresh)

    def _get_locators(self, locator, at, upto, count, every, between, minor):
        log_base, symlog_thresh = self._parse_for_log_params(self.trans)
        if locator is not None:
            major_locator = locator
        elif upto is not None:
            if log_base:
                major_locator = LogLocator(base=log_base, numticks=upto)
            else:
                major_locator = MaxNLocator(upto, steps=[1, 1.5, 2, 2.5, 3, 5, 10])
        elif count is not None:
            if between is None:
                major_locator = LinearLocator(count)
            else:
                if log_base or symlog_thresh:
                    forward, inverse = self._get_transform()
                    lo, hi = forward(between)
                    ticks = inverse(np.linspace(lo, hi, num=count))
                else:
                    ticks = np.linspace(*between, num=count)
                major_locator = FixedLocator(ticks)
        elif every is not None:
            if between is None:
                major_locator = MultipleLocator(every)
            else:
                lo, hi = between
                ticks = np.arange(lo, hi + every, every)
                major_locator = FixedLocator(ticks)
        elif at is not None:
            major_locator = FixedLocator(at)
        elif log_base:
            major_locator = LogLocator(log_base)
        elif symlog_thresh:
            major_locator = SymmetricalLogLocator(linthresh=symlog_thresh, base=10)
        else:
            major_locator = AutoLocator()
        if minor is None:
            minor_locator = LogLocator(log_base, subs=None) if log_base else None
        elif log_base:
            subs = np.linspace(0, log_base, minor + 2)[1:-1]
            minor_locator = LogLocator(log_base, subs=subs)
        else:
            minor_locator = AutoMinorLocator(minor + 1)
        return (major_locator, minor_locator)

    def _get_formatter(self, locator, formatter, like, base, unit):
        log_base, symlog_thresh = self._parse_for_log_params(self.trans)
        if base is default:
            if symlog_thresh:
                log_base = 10
            base = log_base
        if formatter is not None:
            return formatter
        if like is not None:
            if isinstance(like, str):
                if '{x' in like or '{pos' in like:
                    fmt = like
                else:
                    fmt = f'{{x:{like}}}'
                formatter = StrMethodFormatter(fmt)
            else:
                formatter = FuncFormatter(like)
        elif base is not None:
            formatter = LogFormatterSciNotation(base)
        elif unit is not None:
            if isinstance(unit, tuple):
                sep, unit = unit
            elif not unit:
                sep = ''
            else:
                sep = ' '
            formatter = EngFormatter(unit, sep=sep)
        else:
            formatter = ScalarFormatter()
        return formatter