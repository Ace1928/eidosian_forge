import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
class CircleCollection(_CollectionWithSizes):
    """A collection of circles, drawn using splines."""
    _factor = np.pi ** (-1 / 2)

    def __init__(self, sizes, **kwargs):
        """
        Parameters
        ----------
        sizes : float or array-like
            The area of each circle in points^2.
        **kwargs
            Forwarded to `.Collection`.
        """
        super().__init__(**kwargs)
        self.set_sizes(sizes)
        self.set_transform(transforms.IdentityTransform())
        self._paths = [mpath.Path.unit_circle()]