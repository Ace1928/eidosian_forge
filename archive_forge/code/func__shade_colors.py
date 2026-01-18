import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def _shade_colors(color, normals, lightsource=None):
    """
    Shade *color* using normal vectors given by *normals*,
    assuming a *lightsource* (using default position if not given).
    *color* can also be an array of the same length as *normals*.
    """
    if lightsource is None:
        lightsource = mcolors.LightSource(azdeg=225, altdeg=19.4712)
    with np.errstate(invalid='ignore'):
        shade = normals / np.linalg.norm(normals, axis=1, keepdims=True) @ lightsource.direction
    mask = ~np.isnan(shade)
    if mask.any():
        in_norm = mcolors.Normalize(-1, 1)
        out_norm = mcolors.Normalize(0.3, 1).inverse

        def norm(x):
            return out_norm(in_norm(x))
        shade[~mask] = 0
        color = mcolors.to_rgba_array(color)
        alpha = color[:, 3]
        colors = norm(shade)[:, np.newaxis] * color
        colors[:, 3] = alpha
    else:
        colors = np.asanyarray(color).copy()
    return colors