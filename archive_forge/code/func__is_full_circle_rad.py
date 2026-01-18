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
def _is_full_circle_rad(thetamin, thetamax):
    """
    Determine if a wedge (in radians) spans the full circle.

    The condition is derived from :class:`~matplotlib.patches.Wedge`.
    """
    return abs(abs(thetamax - thetamin) - 2 * np.pi) < 1.74e-14