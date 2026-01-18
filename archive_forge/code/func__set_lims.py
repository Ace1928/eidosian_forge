import numbers
import numpy as np
from matplotlib import _api, _docstring
import matplotlib.ticker as mticker
from matplotlib.axes._base import _AxesBase, _TransformedBoundsLocator
from matplotlib.axis import Axis
def _set_lims(self):
    """
        Set the limits based on parent limits and the convert method
        between the parent and this secondary axes.
        """
    if self._orientation == 'x':
        lims = self._parent.get_xlim()
        set_lim = self.set_xlim
    else:
        lims = self._parent.get_ylim()
        set_lim = self.set_ylim
    order = lims[0] < lims[1]
    lims = self._functions[0](np.array(lims))
    neworder = lims[0] < lims[1]
    if neworder != order:
        lims = lims[::-1]
    set_lim(lims)