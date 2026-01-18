import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
class Collection3D(Collection):
    """A collection of 3D paths."""

    def do_3d_projection(self):
        """Project the points according to renderer matrix."""
        xyzs_list = [proj3d.proj_transform(*vs.T, self.axes.M) for vs, _ in self._3dverts_codes]
        self._paths = [mpath.Path(np.column_stack([xs, ys]), cs) for (xs, ys, _), (_, cs) in zip(xyzs_list, self._3dverts_codes)]
        zs = np.concatenate([zs for _, _, zs in xyzs_list])
        return zs.min() if len(zs) else 1000000000.0