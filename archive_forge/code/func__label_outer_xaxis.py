from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def _label_outer_xaxis(self, *, skip_non_rectangular_axes, remove_inner_ticks=False):
    if skip_non_rectangular_axes and (not isinstance(self.patch, mpl.patches.Rectangle)):
        return
    ss = self.get_subplotspec()
    if not ss:
        return
    label_position = self.xaxis.get_label_position()
    if not ss.is_first_row():
        if label_position == 'top':
            self.set_xlabel('')
        top_kw = {'top': False} if remove_inner_ticks else {}
        self.xaxis.set_tick_params(which='both', labeltop=False, **top_kw)
        if self.xaxis.offsetText.get_position()[1] == 1:
            self.xaxis.offsetText.set_visible(False)
    if not ss.is_last_row():
        if label_position == 'bottom':
            self.set_xlabel('')
        bottom_kw = {'bottom': False} if remove_inner_ticks else {}
        self.xaxis.set_tick_params(which='both', labelbottom=False, **bottom_kw)
        if self.xaxis.offsetText.get_position()[1] == 0:
            self.xaxis.offsetText.set_visible(False)