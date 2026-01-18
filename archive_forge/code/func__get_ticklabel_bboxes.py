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
def _get_ticklabel_bboxes(self, ticks, renderer=None):
    """Return lists of bboxes for ticks' label1's and label2's."""
    if renderer is None:
        renderer = self.figure._get_renderer()
    return ([tick.label1.get_window_extent(renderer) for tick in ticks if tick.label1.get_visible()], [tick.label2.get_window_extent(renderer) for tick in ticks if tick.label2.get_visible()])