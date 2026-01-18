import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
class LogLocator(Locator):
    """

    Determine the tick locations for log axes.

    Place ticks on the locations : ``subs[j] * base**i``

    Parameters
    ----------
    base : float, default: 10.0
        The base of the log used, so major ticks are placed at
        ``base**n``, where ``n`` is an integer.
    subs : None or {'auto', 'all'} or sequence of float, default: (1.0,)
        Gives the multiples of integer powers of the base at which
        to place ticks.  The default of ``(1.0, )`` places ticks only at
        integer powers of the base.
        Permitted string values are ``'auto'`` and ``'all'``.
        Both of these use an algorithm based on the axis view
        limits to determine whether and how to put ticks between
        integer powers of the base.  With ``'auto'``, ticks are
        placed only between integer powers; with ``'all'``, the
        integer powers are included.  A value of None is
        equivalent to ``'auto'``.
    numticks : None or int, default: None
        The maximum number of ticks to allow on a given axis. The default
        of ``None`` will try to choose intelligently as long as this
        Locator has already been assigned to an axis using
        `~.axis.Axis.get_tick_space`, but otherwise falls back to 9.

    """

    @_api.delete_parameter('3.8', 'numdecs')
    def __init__(self, base=10.0, subs=(1.0,), numdecs=4, numticks=None):
        """Place ticks on the locations : subs[j] * base**i."""
        if numticks is None:
            if mpl.rcParams['_internal.classic_mode']:
                numticks = 15
            else:
                numticks = 'auto'
        self._base = float(base)
        self._set_subs(subs)
        self._numdecs = numdecs
        self.numticks = numticks

    @_api.delete_parameter('3.8', 'numdecs')
    def set_params(self, base=None, subs=None, numdecs=None, numticks=None):
        """Set parameters within this locator."""
        if base is not None:
            self._base = float(base)
        if subs is not None:
            self._set_subs(subs)
        if numdecs is not None:
            self._numdecs = numdecs
        if numticks is not None:
            self.numticks = numticks
    numdecs = _api.deprecate_privatize_attribute('3.8', addendum='This attribute has no effect.')

    def _set_subs(self, subs):
        """
        Set the minor ticks for the log scaling every ``base**i*subs[j]``.
        """
        if subs is None:
            self._subs = 'auto'
        elif isinstance(subs, str):
            _api.check_in_list(('all', 'auto'), subs=subs)
            self._subs = subs
        else:
            try:
                self._subs = np.asarray(subs, dtype=float)
            except ValueError as e:
                raise ValueError(f"subs must be None, 'all', 'auto' or a sequence of floats, not {subs}.") from e
            if self._subs.ndim != 1:
                raise ValueError(f'A sequence passed to subs must be 1-dimensional, not {self._subs.ndim}-dimensional.')

    def __call__(self):
        """Return the locations of the ticks."""
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if self.numticks == 'auto':
            if self.axis is not None:
                numticks = np.clip(self.axis.get_tick_space(), 2, 9)
            else:
                numticks = 9
        else:
            numticks = self.numticks
        b = self._base
        if vmin <= 0.0:
            if self.axis is not None:
                vmin = self.axis.get_minpos()
            if vmin <= 0.0 or not np.isfinite(vmin):
                raise ValueError('Data has no positive values, and therefore cannot be log-scaled.')
        _log.debug('vmin %s vmax %s', vmin, vmax)
        if vmax < vmin:
            vmin, vmax = (vmax, vmin)
        log_vmin = math.log(vmin) / math.log(b)
        log_vmax = math.log(vmax) / math.log(b)
        numdec = math.floor(log_vmax) - math.ceil(log_vmin)
        if isinstance(self._subs, str):
            if numdec > 10 or b < 3:
                if self._subs == 'auto':
                    return np.array([])
                else:
                    subs = np.array([1.0])
            else:
                _first = 2.0 if self._subs == 'auto' else 1.0
                subs = np.arange(_first, b)
        else:
            subs = self._subs
        stride = max(math.ceil(numdec / (numticks - 1)), 1) if mpl.rcParams['_internal.classic_mode'] else numdec // numticks + 1
        if stride >= numdec:
            stride = max(1, numdec - 1)
        have_subs = len(subs) > 1 or (len(subs) == 1 and subs[0] != 1.0)
        decades = np.arange(math.floor(log_vmin) - stride, math.ceil(log_vmax) + 2 * stride, stride)
        if have_subs:
            if stride == 1:
                ticklocs = np.concatenate([subs * decade_start for decade_start in b ** decades])
            else:
                ticklocs = np.array([])
        else:
            ticklocs = b ** decades
        _log.debug('ticklocs %r', ticklocs)
        if len(subs) > 1 and stride == 1 and (((vmin <= ticklocs) & (ticklocs <= vmax)).sum() <= 1):
            return AutoLocator().tick_values(vmin, vmax)
        else:
            return self.raise_if_exceeds(ticklocs)

    def view_limits(self, vmin, vmax):
        """Try to choose the view limits intelligently."""
        b = self._base
        vmin, vmax = self.nonsingular(vmin, vmax)
        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            vmin = _decade_less_equal(vmin, b)
            vmax = _decade_greater_equal(vmax, b)
        return (vmin, vmax)

    def nonsingular(self, vmin, vmax):
        if vmin > vmax:
            vmin, vmax = (vmax, vmin)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = (1, 10)
        elif vmax <= 0:
            _api.warn_external('Data has no positive values, and therefore cannot be log-scaled.')
            vmin, vmax = (1, 10)
        else:
            minpos = min((axis.get_minpos() for axis in self.axis._get_shared_axis()))
            if not np.isfinite(minpos):
                minpos = 1e-300
            if vmin <= 0:
                vmin = minpos
            if vmin == vmax:
                vmin = _decade_less(vmin, self._base)
                vmax = _decade_greater(vmax, self._base)
        return (vmin, vmax)