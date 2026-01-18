import math
import numpy as np
import matplotlib as mpl
from matplotlib.patches import _Style, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform
class FilledArrow(SimpleArrow):
    """
        An arrow with a filled head.
        """
    ArrowAxisClass = _FancyAxislineStyle.FilledArrow

    def __init__(self, size=1, facecolor=None):
        """
            Parameters
            ----------
            size : float
                Size of the arrow as a fraction of the ticklabel size.
            facecolor : color, default: :rc:`axes.edgecolor`
                Fill color.

                .. versionadded:: 3.7
            """
        if facecolor is None:
            facecolor = mpl.rcParams['axes.edgecolor']
        self.size = size
        self._facecolor = facecolor
        super().__init__(size=size)

    def new_line(self, axis_artist, transform):
        linepath = Path([(0, 0), (0, 1)])
        axisline = self.ArrowAxisClass(axis_artist, linepath, transform, line_mutation_scale=self.size, facecolor=self._facecolor)
        return axisline