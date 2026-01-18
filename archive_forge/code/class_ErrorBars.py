import numpy as np
import param
from ..core import Dataset, Dimension, Element2D, NdOverlay, Overlay, util
from ..core.dimension import process_dimensions
from .geom import (  # noqa: F401 backward compatible import
from .selection import Selection1DExpr
class ErrorBars(Selection1DExpr, Chart):
    """
    ErrorBars is a Chart element representing error bars in a 1D
    coordinate system where the key dimension corresponds to the
    location along the x-axis and the first value dimension
    corresponds to the location along the y-axis and one or two
    extra value dimensions corresponding to the symmetric or
    asymmetric errors either along x-axis or y-axis. If two value
    dimensions are given, then the last value dimension will be
    taken as symmetric errors. If three value dimensions are given
    then the last two value dimensions will be taken as negative and
    positive errors. By default the errors are defined along y-axis.
    A parameter `horizontal`, when set `True`, will define the errors
    along the x-axis.
    """
    group = param.String(default='ErrorBars', constant=True, doc='\n        A string describing the quantity measured by the ErrorBars\n        object.')
    vdims = param.List(default=[Dimension('y'), Dimension('yerror')], bounds=(1, None), constant=True)
    horizontal = param.Boolean(default=False, doc='\n        Whether the errors are along y-axis (vertical) or x-axis.')

    def range(self, dim, data_range=True, dimension_range=True):
        """Return the lower and upper bounds of values along dimension.

        Range of the y-dimension includes the symmetric or asymmetric
        error.

        Args:
            dimension: The dimension to compute the range on.
            data_range (bool): Compute range from data values
            dimension_range (bool): Include Dimension ranges
                Whether to include Dimension range and soft_range
                in range calculation

        Returns:
            Tuple containing the lower and upper bound
        """
        dim_with_err = 0 if self.horizontal else 1
        didx = self.get_dimension_index(dim)
        dim = self.get_dimension(dim)
        if didx == dim_with_err and data_range and len(self):
            mean = self.dimension_values(didx)
            neg_error = self.dimension_values(2)
            if len(self.dimensions()) > 3:
                pos_error = self.dimension_values(3)
            else:
                pos_error = neg_error
            lower = np.nanmin(mean - neg_error)
            upper = np.nanmax(mean + pos_error)
            if not dimension_range:
                return (lower, upper)
            return util.dimension_range(lower, upper, dim.range, dim.soft_range)
        return super().range(dim, data_range)