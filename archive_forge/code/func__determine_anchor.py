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
def _determine_anchor(self, mode, angle, start):
    if mode == 'auto':
        if start:
            if -90 <= angle <= 90:
                return ('left', 'center')
            else:
                return ('right', 'center')
        elif -90 <= angle <= 90:
            return ('right', 'center')
        else:
            return ('left', 'center')
    elif start:
        if angle < -68.5:
            return ('center', 'top')
        elif angle < -23.5:
            return ('left', 'top')
        elif angle < 22.5:
            return ('left', 'center')
        elif angle < 67.5:
            return ('left', 'bottom')
        elif angle < 112.5:
            return ('center', 'bottom')
        elif angle < 157.5:
            return ('right', 'bottom')
        elif angle < 202.5:
            return ('right', 'center')
        elif angle < 247.5:
            return ('right', 'top')
        else:
            return ('center', 'top')
    elif angle < -68.5:
        return ('center', 'bottom')
    elif angle < -23.5:
        return ('right', 'bottom')
    elif angle < 22.5:
        return ('right', 'center')
    elif angle < 67.5:
        return ('right', 'top')
    elif angle < 112.5:
        return ('center', 'top')
    elif angle < 157.5:
        return ('left', 'top')
    elif angle < 202.5:
        return ('left', 'center')
    elif angle < 247.5:
        return ('left', 'bottom')
    else:
        return ('center', 'bottom')