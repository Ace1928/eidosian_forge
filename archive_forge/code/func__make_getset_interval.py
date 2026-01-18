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
def _make_getset_interval(method_name, lim_name, attr_name):
    """
    Helper to generate ``get_{data,view}_interval`` and
    ``set_{data,view}_interval`` implementations.
    """

    def getter(self):
        return getattr(getattr(self.axes, lim_name), attr_name)

    def setter(self, vmin, vmax, ignore=False):
        if ignore:
            setattr(getattr(self.axes, lim_name), attr_name, (vmin, vmax))
        else:
            oldmin, oldmax = getter(self)
            if oldmin < oldmax:
                setter(self, min(vmin, vmax, oldmin), max(vmin, vmax, oldmax), ignore=True)
            else:
                setter(self, max(vmin, vmax, oldmin), min(vmin, vmax, oldmax), ignore=True)
        self.stale = True
    getter.__name__ = f'get_{method_name}_interval'
    setter.__name__ = f'set_{method_name}_interval'
    return (getter, setter)