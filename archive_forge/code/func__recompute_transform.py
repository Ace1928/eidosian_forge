from collections.abc import MutableMapping
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath
def _recompute_transform(self):
    """
        Notes
        -----
        This cannot be called until after this has been added to an Axes,
        otherwise unit conversion will fail. This makes it very important to
        call the accessor method and not directly access the transformation
        member variable.
        """
    assert self._patch_type in ('arc', 'circle')
    center = (self.convert_xunits(self._center[0]), self.convert_yunits(self._center[1]))
    width = self.convert_xunits(self._width)
    height = self.convert_yunits(self._height)
    self._patch_transform = mtransforms.Affine2D().scale(width * 0.5, height * 0.5).translate(*center)