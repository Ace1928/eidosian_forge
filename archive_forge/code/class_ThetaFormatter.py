import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.spines as mspines
from matplotlib.ticker import (
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
class ThetaFormatter(Formatter):
    """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """

    def __init__(self, round_to=1.0):
        self._round_to = round_to

    def __call__(self, x, pos=None):
        degrees = round(np.rad2deg(x) / self._round_to) * self._round_to
        return f'{degrees:0.0f}°'