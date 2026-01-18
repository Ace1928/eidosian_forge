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
def _set_title_offset_trans(self, title_offset_points):
    """
        Set the offset for the title either from :rc:`axes.titlepad`
        or from set_title kwarg ``pad``.
        """
    self.titleOffsetTrans = mtransforms.ScaledTranslation(0.0, title_offset_points / 72, self.figure.dpi_scale_trans)
    for _title in (self.title, self._left_title, self._right_title):
        _title.set_transform(self.transAxes + self.titleOffsetTrans)
        _title.set_clip_box(None)