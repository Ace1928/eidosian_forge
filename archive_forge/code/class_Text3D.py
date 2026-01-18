import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
class Text3D(mtext.Text):
    """
    Text object with 3D position and direction.

    Parameters
    ----------
    x, y, z : float
        The position of the text.
    text : str
        The text string to display.
    zdir : {'x', 'y', 'z', None, 3-tuple}
        The direction of the text. See `.get_dir_vector` for a description of
        the values.

    Other Parameters
    ----------------
    **kwargs
         All other parameters are passed on to `~matplotlib.text.Text`.
    """

    def __init__(self, x=0, y=0, z=0, text='', zdir='z', **kwargs):
        mtext.Text.__init__(self, x, y, text, **kwargs)
        self.set_3d_properties(z, zdir)

    def get_position_3d(self):
        """Return the (x, y, z) position of the text."""
        return (self._x, self._y, self._z)

    def set_position_3d(self, xyz, zdir=None):
        """
        Set the (*x*, *y*, *z*) position of the text.

        Parameters
        ----------
        xyz : (float, float, float)
            The position in 3D space.
        zdir : {'x', 'y', 'z', None, 3-tuple}
            The direction of the text. If unspecified, the *zdir* will not be
            changed. See `.get_dir_vector` for a description of the values.
        """
        super().set_position(xyz[:2])
        self.set_z(xyz[2])
        if zdir is not None:
            self._dir_vec = get_dir_vector(zdir)

    def set_z(self, z):
        """
        Set the *z* position of the text.

        Parameters
        ----------
        z : float
        """
        self._z = z
        self.stale = True

    def set_3d_properties(self, z=0, zdir='z'):
        """
        Set the *z* position and direction of the text.

        Parameters
        ----------
        z : float
            The z-position in 3D space.
        zdir : {'x', 'y', 'z', 3-tuple}
            The direction of the text. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        """
        self._z = z
        self._dir_vec = get_dir_vector(zdir)
        self.stale = True

    @artist.allow_rasterization
    def draw(self, renderer):
        position3d = np.array((self._x, self._y, self._z))
        proj = proj3d._proj_trans_points([position3d, position3d + self._dir_vec], self.axes.M)
        dx = proj[0][1] - proj[0][0]
        dy = proj[1][1] - proj[1][0]
        angle = math.degrees(math.atan2(dy, dx))
        with cbook._setattr_cm(self, _x=proj[0][0], _y=proj[1][0], _rotation=_norm_text_angle(angle)):
            mtext.Text.draw(self, renderer)
        self.stale = False

    def get_tightbbox(self, renderer=None):
        return None