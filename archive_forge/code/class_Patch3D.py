import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
class Patch3D(Patch):
    """
    3D patch object.
    """

    def __init__(self, *args, zs=(), zdir='z', **kwargs):
        """
        Parameters
        ----------
        verts :
        zs : float
            The location along the *zdir* axis in 3D space to position the
            patch.
        zdir : {'x', 'y', 'z'}
            Plane to plot patch orthogonal to. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        """
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir)

    def set_3d_properties(self, verts, zs=0, zdir='z'):
        """
        Set the *z* position and direction of the patch.

        Parameters
        ----------
        verts :
        zs : float
            The location along the *zdir* axis in 3D space to position the
            patch.
        zdir : {'x', 'y', 'z'}
            Plane to plot patch orthogonal to. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        """
        zs = np.broadcast_to(zs, len(verts))
        self._segment3d = [juggle_axes(x, y, z, zdir) for (x, y), z in zip(verts, zs)]

    def get_path(self):
        return self._path2d

    def do_3d_projection(self):
        s = self._segment3d
        xs, ys, zs = zip(*s)
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs, self.axes.M)
        self._path2d = mpath.Path(np.column_stack([vxs, vys]))
        return min(vzs)