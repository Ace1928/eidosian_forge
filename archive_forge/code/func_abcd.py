from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
@property
def abcd(self):
    """Return a tuple with the plane coefficients.

        Returns:
            Tuple with the plane coefficients.
        """
    return (self._coefficients[0], self._coefficients[1], self._coefficients[2], self._coefficients[3])