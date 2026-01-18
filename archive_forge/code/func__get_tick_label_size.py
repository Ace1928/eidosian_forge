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
def _get_tick_label_size(self, axis_name):
    """
        Return the text size of tick labels for this Axis.

        This is a convenience function to avoid having to create a `Tick` in
        `.get_tick_space`, since it is expensive.
        """
    tick_kw = self._major_tick_kw
    size = tick_kw.get('labelsize', mpl.rcParams[f'{axis_name}tick.labelsize'])
    return mtext.FontProperties(size=size).get_size_in_points()