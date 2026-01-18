import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
@staticmethod
def _convert_mesh_to_paths(coordinates):
    """
        Convert a given mesh into a sequence of `.Path` objects.

        This function is primarily of use to implementers of backends that do
        not directly support quadmeshes.
        """
    if isinstance(coordinates, np.ma.MaskedArray):
        c = coordinates.data
    else:
        c = coordinates
    points = np.concatenate([c[:-1, :-1], c[:-1, 1:], c[1:, 1:], c[1:, :-1], c[:-1, :-1]], axis=2).reshape((-1, 5, 2))
    return [mpath.Path(x) for x in points]