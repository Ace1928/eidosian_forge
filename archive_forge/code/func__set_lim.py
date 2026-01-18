import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
def _set_lim(self, v0, v1, *, emit=True, auto):
    """
        Set view limits.

        This method is a helper for the Axes ``set_xlim``, ``set_ylim``, and
        ``set_zlim`` methods.

        Parameters
        ----------
        v0, v1 : float
            The view limits.  (Passing *v0* as a (low, high) pair is not
            supported; normalization must occur in the Axes setters.)
        emit : bool, default: True
            Whether to notify observers of limit change.
        auto : bool or None, default: False
            Whether to turn on autoscaling of the x-axis. True turns on, False
            turns off, None leaves unchanged.
        """
    name = self._get_axis_name()
    self.axes._process_unit_info([(name, (v0, v1))], convert=False)
    v0 = self.axes._validate_converted_limits(v0, self.convert_units)
    v1 = self.axes._validate_converted_limits(v1, self.convert_units)
    if v0 is None or v1 is None:
        old0, old1 = self.get_view_interval()
        if v0 is None:
            v0 = old0
        if v1 is None:
            v1 = old1
    if self.get_scale() == 'log' and (v0 <= 0 or v1 <= 0):
        old0, old1 = self.get_view_interval()
        if v0 <= 0:
            _api.warn_external(f'Attempt to set non-positive {name}lim on a log-scaled axis will be ignored.')
            v0 = old0
        if v1 <= 0:
            _api.warn_external(f'Attempt to set non-positive {name}lim on a log-scaled axis will be ignored.')
            v1 = old1
    if v0 == v1:
        _api.warn_external(f'Attempting to set identical low and high {name}lims makes transformation singular; automatically expanding.')
    reverse = bool(v0 > v1)
    v0, v1 = self.get_major_locator().nonsingular(v0, v1)
    v0, v1 = self.limit_range_for_scale(v0, v1)
    v0, v1 = sorted([v0, v1], reverse=bool(reverse))
    self.set_view_interval(v0, v1, ignore=True)
    for ax in self._get_shared_axes():
        ax._stale_viewlims[name] = False
    if auto is not None:
        self._set_autoscale_on(bool(auto))
    if emit:
        self.axes.callbacks.process(f'{name}lim_changed', self.axes)
        for other in self._get_shared_axes():
            if other is self.axes:
                continue
            other._axis_map[name]._set_lim(v0, v1, emit=False, auto=auto)
            if emit:
                other.callbacks.process(f'{name}lim_changed', other)
            if other.figure != self.figure:
                other.figure.canvas.draw_idle()
    self.stale = True
    return (v0, v1)