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
def _apply_tickdir(self, tickdir):
    super()._apply_tickdir(tickdir)
    mark1, mark2 = {'out': (mlines.TICKLEFT, mlines.TICKRIGHT), 'in': (mlines.TICKRIGHT, mlines.TICKLEFT), 'inout': ('_', '_')}[self._tickdir]
    self.tick1line.set_marker(mark1)
    self.tick2line.set_marker(mark2)