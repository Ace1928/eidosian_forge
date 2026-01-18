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
def _set_lim_and_transforms(self):
    """
        Set the *_xaxis_transform*, *_yaxis_transform*, *transScale*,
        *transData*, *transLimits* and *transAxes* transformations.

        .. note::

            This method is primarily used by rectilinear projections of the
            `~matplotlib.axes.Axes` class, and is meant to be overridden by
            new kinds of projection Axes that need different transformations
            and limits. (See `~matplotlib.projections.polar.PolarAxes` for an
            example.)
        """
    self.transAxes = mtransforms.BboxTransformTo(self.bbox)
    self.transScale = mtransforms.TransformWrapper(mtransforms.IdentityTransform())
    self.transLimits = mtransforms.BboxTransformFrom(mtransforms.TransformedBbox(self._viewLim, self.transScale))
    self.transData = self.transScale + (self.transLimits + self.transAxes)
    self._xaxis_transform = mtransforms.blended_transform_factory(self.transData, self.transAxes)
    self._yaxis_transform = mtransforms.blended_transform_factory(self.transAxes, self.transData)