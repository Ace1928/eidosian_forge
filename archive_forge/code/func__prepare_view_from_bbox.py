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
def _prepare_view_from_bbox(self, bbox, direction='in', mode=None, twinx=False, twiny=False):
    """
        Helper function to prepare the new bounds from a bbox.

        This helper function returns the new x and y bounds from the zoom
        bbox. This a convenience method to abstract the bbox logic
        out of the base setter.
        """
    if len(bbox) == 3:
        xp, yp, scl = bbox
        if scl == 0:
            scl = 1.0
        if scl > 1:
            direction = 'in'
        else:
            direction = 'out'
            scl = 1 / scl
        (xmin, ymin), (xmax, ymax) = self.transData.transform(np.transpose([self.get_xlim(), self.get_ylim()]))
        xwidth = xmax - xmin
        ywidth = ymax - ymin
        xcen = (xmax + xmin) * 0.5
        ycen = (ymax + ymin) * 0.5
        xzc = (xp * (scl - 1) + xcen) / scl
        yzc = (yp * (scl - 1) + ycen) / scl
        bbox = [xzc - xwidth / 2.0 / scl, yzc - ywidth / 2.0 / scl, xzc + xwidth / 2.0 / scl, yzc + ywidth / 2.0 / scl]
    elif len(bbox) != 4:
        _api.warn_external('Warning in _set_view_from_bbox: bounding box is not a tuple of length 3 or 4. Ignoring the view change.')
        return
    xmin0, xmax0 = self.get_xbound()
    ymin0, ymax0 = self.get_ybound()
    startx, starty, stopx, stopy = bbox
    (startx, starty), (stopx, stopy) = self.transData.inverted().transform([(startx, starty), (stopx, stopy)])
    xmin, xmax = np.clip(sorted([startx, stopx]), xmin0, xmax0)
    ymin, ymax = np.clip(sorted([starty, stopy]), ymin0, ymax0)
    if twinx or mode == 'y':
        xmin, xmax = (xmin0, xmax0)
    if twiny or mode == 'x':
        ymin, ymax = (ymin0, ymax0)
    if direction == 'in':
        new_xbound = (xmin, xmax)
        new_ybound = (ymin, ymax)
    elif direction == 'out':
        x_trf = self.xaxis.get_transform()
        sxmin0, sxmax0, sxmin, sxmax = x_trf.transform([xmin0, xmax0, xmin, xmax])
        factor = (sxmax0 - sxmin0) / (sxmax - sxmin)
        sxmin1 = sxmin0 - factor * (sxmin - sxmin0)
        sxmax1 = sxmax0 + factor * (sxmax0 - sxmax)
        new_xbound = x_trf.inverted().transform([sxmin1, sxmax1])
        y_trf = self.yaxis.get_transform()
        symin0, symax0, symin, symax = y_trf.transform([ymin0, ymax0, ymin, ymax])
        factor = (symax0 - symin0) / (symax - symin)
        symin1 = symin0 - factor * (symin - symin0)
        symax1 = symax0 + factor * (symax0 - symax)
        new_ybound = y_trf.inverted().transform([symin1, symax1])
    return (new_xbound, new_ybound)