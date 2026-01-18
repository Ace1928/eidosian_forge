from numbers import Number
import numpy as np
import param
from ..core import Dimension, Element, Element2D
from ..core.data import Dataset
from ..core.util import datetime_types
class Spline(Annotation):
    """
    Draw a spline using the given handle coordinates and handle
    codes. The constructor accepts a tuple in format (coords, codes).

    Follows format of matplotlib spline definitions as used in
    matplotlib.path.Path with the following codes:

    Path.STOP     : 0
    Path.MOVETO   : 1
    Path.LINETO   : 2
    Path.CURVE3   : 3
    Path.CURVE4   : 4
    Path.CLOSEPLOY: 79
    """
    group = param.String(default='Spline', constant=True)

    def __init__(self, spline_points, **params):
        super().__init__(spline_points, **params)

    def clone(self, data=None, shared_data=True, new_type=None, *args, **overrides):
        """Clones the object, overriding data and parameters.

        Args:
            data: New data replacing the existing data
            shared_data (bool, optional): Whether to use existing data
            new_type (optional): Type to cast object to
            *args: Additional arguments to pass to constructor
            **overrides: New keyword arguments to pass to constructor

        Returns:
            Cloned Spline
        """
        return Element2D.clone(self, data, shared_data, new_type, *args, **overrides)

    def dimension_values(self, dimension, expanded=True, flat=True):
        """Return the values along the requested dimension.

        Args:
            dimension: The dimension to return values for
            expanded (bool, optional): Whether to expand values
            flat (bool, optional): Whether to flatten array

        Returns:
            NumPy array of values along the requested dimension
        """
        index = self.get_dimension_index(dimension)
        if index in [0, 1]:
            return np.array([point[index] for point in self.data[0]])
        else:
            return super().dimension_values(dimension)