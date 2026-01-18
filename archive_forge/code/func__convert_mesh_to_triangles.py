import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def _convert_mesh_to_triangles(self, coordinates):
    """
        Convert a given mesh into a sequence of triangles, each point
        with its own color.  The result can be used to construct a call to
        `~.RendererBase.draw_gouraud_triangles`.
        """
    if isinstance(coordinates, np.ma.MaskedArray):
        p = coordinates.data
    else:
        p = coordinates
    p_a = p[:-1, :-1]
    p_b = p[:-1, 1:]
    p_c = p[1:, 1:]
    p_d = p[1:, :-1]
    p_center = (p_a + p_b + p_c + p_d) / 4.0
    triangles = np.concatenate([p_a, p_b, p_center, p_b, p_c, p_center, p_c, p_d, p_center, p_d, p_a, p_center], axis=2).reshape((-1, 3, 2))
    c = self.get_facecolor().reshape((*coordinates.shape[:2], 4))
    z = self.get_array()
    mask = z.mask if np.ma.is_masked(z) else None
    if mask is not None:
        c[mask, 3] = np.nan
    c_a = c[:-1, :-1]
    c_b = c[:-1, 1:]
    c_c = c[1:, 1:]
    c_d = c[1:, :-1]
    c_center = (c_a + c_b + c_c + c_d) / 4.0
    colors = np.concatenate([c_a, c_b, c_center, c_b, c_c, c_center, c_c, c_d, c_center, c_d, c_a, c_center], axis=2).reshape((-1, 3, 4))
    tmask = np.isnan(colors[..., 2, 3])
    return (triangles[~tmask], colors[~tmask])