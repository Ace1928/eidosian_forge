from collections.abc import Iterable
from io import StringIO
from numbers import Integral
import numpy as np
from ..base import is_classifier
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from ..utils.validation import check_array, check_is_fitted
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim
def _color_brew(n):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []
    s, v = (0.75, 0.9)
    c = s * v
    m = v - c
    for h in np.arange(25, 385, 360.0 / n).astype(int):
        h_bar = h / 60.0
        x = c * (1 - abs(h_bar % 2 - 1))
        rgb = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x), (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        rgb = [int(255 * (r + m)), int(255 * (g + m)), int(255 * (b + m))]
        color_list.append(rgb)
    return color_list