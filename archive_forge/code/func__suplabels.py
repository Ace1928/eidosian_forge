from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
def _suplabels(self, t, info, **kwargs):
    """
        Add a centered %(name)s to the figure.

        Parameters
        ----------
        t : str
            The %(name)s text.
        x : float, default: %(x0)s
            The x location of the text in figure coordinates.
        y : float, default: %(y0)s
            The y location of the text in figure coordinates.
        horizontalalignment, ha : {'center', 'left', 'right'}, default: %(ha)s
            The horizontal alignment of the text relative to (*x*, *y*).
        verticalalignment, va : {'top', 'center', 'bottom', 'baseline'}, default: %(va)s
            The vertical alignment of the text relative to (*x*, *y*).
        fontsize, size : default: :rc:`figure.%(rc)ssize`
            The font size of the text. See `.Text.set_size` for possible
            values.
        fontweight, weight : default: :rc:`figure.%(rc)sweight`
            The font weight of the text. See `.Text.set_weight` for possible
            values.

        Returns
        -------
        text
            The `.Text` instance of the %(name)s.

        Other Parameters
        ----------------
        fontproperties : None or dict, optional
            A dict of font properties. If *fontproperties* is given the
            default values for font size and weight are taken from the
            `.FontProperties` defaults. :rc:`figure.%(rc)ssize` and
            :rc:`figure.%(rc)sweight` are ignored in this case.

        **kwargs
            Additional kwargs are `matplotlib.text.Text` properties.
        """
    suplab = getattr(self, info['name'])
    x = kwargs.pop('x', None)
    y = kwargs.pop('y', None)
    if info['name'] in ['_supxlabel', '_suptitle']:
        autopos = y is None
    elif info['name'] == '_supylabel':
        autopos = x is None
    if x is None:
        x = info['x0']
    if y is None:
        y = info['y0']
    if 'horizontalalignment' not in kwargs and 'ha' not in kwargs:
        kwargs['horizontalalignment'] = info['ha']
    if 'verticalalignment' not in kwargs and 'va' not in kwargs:
        kwargs['verticalalignment'] = info['va']
    if 'rotation' not in kwargs:
        kwargs['rotation'] = info['rotation']
    if 'fontproperties' not in kwargs:
        if 'fontsize' not in kwargs and 'size' not in kwargs:
            kwargs['size'] = mpl.rcParams[info['size']]
        if 'fontweight' not in kwargs and 'weight' not in kwargs:
            kwargs['weight'] = mpl.rcParams[info['weight']]
    sup = self.text(x, y, t, **kwargs)
    if suplab is not None:
        suplab.set_text(t)
        suplab.set_position((x, y))
        suplab.update_from(sup)
        sup.remove()
    else:
        suplab = sup
    suplab._autopos = autopos
    setattr(self, info['name'], suplab)
    self.stale = True
    return suplab