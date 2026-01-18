import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
class FixedLocator:

    def __init__(self, locs):
        self._locs = locs

    def __call__(self, v1, v2):
        v1, v2 = sorted([v1, v2])
        locs = np.array([l for l in self._locs if v1 <= l <= v2])
        return (locs, len(locs), 1)