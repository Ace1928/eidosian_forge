import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def __equalize_densities(self, nominal_bounds, nominal_density):
    """
        Calculate the true density along x, and adjust the top and
        bottom bounds so that the density along y will be equal.

        Returns (adjusted_bounds, true_density)
        """
    left, bottom, right, top = nominal_bounds.lbrt()
    width, height = (right - left, top - bottom)
    center_y = bottom + height / 2.0
    true_density = int(nominal_density * width) / float(width)
    n_cells = round(height * true_density, 0)
    adjusted_half_height = n_cells / true_density / 2.0
    return (BoundingBox(points=((left, center_y - adjusted_half_height), (right, center_y + adjusted_half_height))), true_density)