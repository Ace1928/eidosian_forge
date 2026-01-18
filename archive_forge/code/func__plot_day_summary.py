from __future__ import (absolute_import, division, print_function,
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from six.moves import xrange, zip
def _plot_day_summary(ax, quotes, ticksize=3, colorup='k', colordown='r', ochl=True):
    """Plots day summary


        Represent the time, open, high, low, close as a vertical line
        ranging from low to high.  The left tick is the open and the right
        tick is the close.



    Parameters
    ----------
    ax : `Axes`
        an `Axes` instance to plot to
    quotes : sequence of quote sequences
        data to plot.  time must be in float date format - see date2num
        (time, open, high, low, close, ...) vs
        (time, open, close, high, low, ...)
        set by `ochl`
    ticksize : int
        open/close tick marker in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    ochl: bool
        argument to select between ochl and ohlc ordering of quotes

    Returns
    -------
    lines : list
        list of tuples of the lines added (one tuple per quote)
    """
    lines = []
    for q in quotes:
        if ochl:
            t, open, close, high, low = q[:5]
        else:
            t, open, high, low, close = q[:5]
        if close >= open:
            color = colorup
        else:
            color = colordown
        vline = Line2D(xdata=(t, t), ydata=(low, high), color=color, antialiased=False)
        oline = Line2D(xdata=(t, t), ydata=(open, open), color=color, antialiased=False, marker=TICKLEFT, markersize=ticksize)
        cline = Line2D(xdata=(t, t), ydata=(close, close), color=color, antialiased=False, markersize=ticksize, marker=TICKRIGHT)
        lines.extend((vline, oline, cline))
        ax.add_line(vline)
        ax.add_line(oline)
        ax.add_line(cline)
    ax.autoscale_view()
    return lines