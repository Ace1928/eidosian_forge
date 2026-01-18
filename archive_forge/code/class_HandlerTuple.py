from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
class HandlerTuple(HandlerBase):
    """
    Handler for Tuple.
    """

    def __init__(self, ndivide=1, pad=None, **kwargs):
        """
        Parameters
        ----------
        ndivide : int or None, default: 1
            The number of sections to divide the legend area into.  If None,
            use the length of the input tuple.
        pad : float, default: :rc:`legend.borderpad`
            Padding in units of fraction of font size.
        **kwargs
            Keyword arguments forwarded to `.HandlerBase`.
        """
        self._ndivide = ndivide
        self._pad = pad
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        handler_map = legend.get_legend_handler_map()
        if self._ndivide is None:
            ndivide = len(orig_handle)
        else:
            ndivide = self._ndivide
        if self._pad is None:
            pad = legend.borderpad * fontsize
        else:
            pad = self._pad * fontsize
        if ndivide > 1:
            width = (width - pad * (ndivide - 1)) / ndivide
        xds_cycle = cycle(xdescent - (width + pad) * np.arange(ndivide))
        a_list = []
        for handle1 in orig_handle:
            handler = legend.get_legend_handler(handler_map, handle1)
            _a_list = handler.create_artists(legend, handle1, next(xds_cycle), ydescent, width, height, fontsize, trans)
            a_list.extend(_a_list)
        return a_list