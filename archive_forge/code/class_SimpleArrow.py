import math
import numpy as np
import matplotlib as mpl
from matplotlib.patches import _Style, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform
class SimpleArrow(_Base):
    """
        A simple arrow.
        """
    ArrowAxisClass = _FancyAxislineStyle.SimpleArrow

    def __init__(self, size=1):
        """
            Parameters
            ----------
            size : float
                Size of the arrow as a fraction of the ticklabel size.
            """
        self.size = size
        super().__init__()

    def new_line(self, axis_artist, transform):
        linepath = Path([(0, 0), (0, 1)])
        axisline = self.ArrowAxisClass(axis_artist, linepath, transform, line_mutation_scale=self.size)
        return axisline