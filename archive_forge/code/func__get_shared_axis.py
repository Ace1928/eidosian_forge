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
def _get_shared_axis(self):
    """Return list of shared axis for current axis."""
    name = self._get_axis_name()
    return [ax._axis_map[name] for ax in self._get_shared_axes()]