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
def add_collection(self, collection, autolim=True):
    """
        Add a `.Collection` to the Axes; return the collection.
        """
    _api.check_isinstance(mcoll.Collection, collection=collection)
    if not collection.get_label():
        collection.set_label(f'_child{len(self._children)}')
    self._children.append(collection)
    collection._remove_method = self._children.remove
    self._set_artist_props(collection)
    if collection.get_clip_path() is None:
        collection.set_clip_path(self.patch)
    if autolim:
        self._unstale_viewLim()
        datalim = collection.get_datalim(self.transData)
        points = datalim.get_points()
        if not np.isinf(datalim.minpos).all():
            points = np.concatenate([points, [datalim.minpos]])
        self.update_datalim(points)
    self.stale = True
    return collection