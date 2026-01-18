from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _single_hsv_to_rgb(hsv):
    """Transform a color from the hsv space to the rgb."""
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(array(hsv).reshape(1, 1, 3)).reshape(3)