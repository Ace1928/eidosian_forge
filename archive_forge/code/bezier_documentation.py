from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api

        Return the dimension and location of the curve's interior extrema.

        The extrema are the points along the curve where one of its partial
        derivatives is zero.

        Returns
        -------
        dims : array of int
            Index :math:`i` of the partial derivative which is zero at each
            interior extrema.
        dzeros : array of float
            Of same size as dims. The :math:`t` such that :math:`d/dx_i B(t) =
            0`
        