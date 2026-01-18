import math
import numpy as np
from numpy import ma
from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms
def _set_transform(self):
    """
        Set the PolyCollection transform to go
        from arrow width units to pixels.
        """
    dx = self._dots_per_unit(self.units)
    self._trans_scale = dx
    trans = transforms.Affine2D().scale(dx)
    self.set_transform(trans)
    return trans