import math
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine
class _ThetaShift(mtransforms.ScaledTranslation):
    """
    Apply a padding shift based on axes theta limits.

    This is used to create padding for radial ticks.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        The owning axes; used to determine limits.
    pad : float
        The padding to apply, in points.
    mode : {'min', 'max', 'rlabel'}
        Whether to shift away from the start (``'min'``) or the end (``'max'``)
        of the axes, or using the rlabel position (``'rlabel'``).
    """

    def __init__(self, axes, pad, mode):
        super().__init__(pad, pad, axes.figure.dpi_scale_trans)
        self.set_children(axes._realViewLim)
        self.axes = axes
        self.mode = mode
        self.pad = pad
    __str__ = mtransforms._make_str_method('axes', 'pad', 'mode')

    def get_matrix(self):
        if self._invalid:
            if self.mode == 'rlabel':
                angle = np.deg2rad(self.axes.get_rlabel_position()) * self.axes.get_theta_direction() + self.axes.get_theta_offset()
            elif self.mode == 'min':
                angle = self.axes._realViewLim.xmin
            elif self.mode == 'max':
                angle = self.axes._realViewLim.xmax
            if self.mode in ('rlabel', 'min'):
                padx = np.cos(angle - np.pi / 2)
                pady = np.sin(angle - np.pi / 2)
            else:
                padx = np.cos(angle + np.pi / 2)
                pady = np.sin(angle + np.pi / 2)
            self._t = (self.pad * padx / 72, self.pad * pady / 72)
        return super().get_matrix()