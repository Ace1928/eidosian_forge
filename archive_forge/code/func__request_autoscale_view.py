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
def _request_autoscale_view(self, axis='all', tight=None):
    """
        Mark a single axis, or all of them, as stale wrt. autoscaling.

        No computation is performed until the next autoscaling; thus, separate
        calls to control individual axises incur negligible performance cost.

        Parameters
        ----------
        axis : str, default: "all"
            Either an element of ``self._axis_names``, or "all".
        tight : bool or None, default: None
        """
    axis_names = _api.check_getitem({**{k: [k] for k in self._axis_names}, 'all': self._axis_names}, axis=axis)
    for name in axis_names:
        self._stale_viewlims[name] = True
    if tight is not None:
        self._tight = tight