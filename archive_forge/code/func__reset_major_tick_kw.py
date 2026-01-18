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
def _reset_major_tick_kw(self):
    self._major_tick_kw.clear()
    self._major_tick_kw['gridOn'] = mpl.rcParams['axes.grid'] and mpl.rcParams['axes.grid.which'] in ('both', 'major')