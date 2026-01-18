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
def _makefill(self, axes, x, y, kw, kwargs):
    x = axes.convert_xunits(x)
    y = axes.convert_yunits(y)
    kw = kw.copy()
    kwargs = kwargs.copy()
    ignores = {'marker', 'markersize', 'markeredgecolor', 'markerfacecolor', 'markeredgewidth'}
    for k, v in kwargs.items():
        if v is not None:
            ignores.add(k)
    default_dict = self._getdefaults(ignores, kw)
    self._setdefaults(default_dict, kw)
    facecolor = kw.get('color', None)
    default_dict.pop('color', None)
    self._setdefaults(default_dict, kwargs)
    seg = mpatches.Polygon(np.column_stack((x, y)), facecolor=facecolor, fill=kwargs.get('fill', True), closed=kw['closed'])
    seg.set(**kwargs)
    return (seg, kwargs)