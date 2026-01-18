import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
class gradient(Operation):
    """
    Compute the gradient plot of the supplied Image.

    If the Image value dimension is cyclic, the smallest step is taken
    considered the cyclic range
    """
    output_type = Image
    group = param.String(default='Gradient', doc='\n    The group assigned to the output gradient matrix.')
    _per_element = True

    def _process(self, matrix, key=None):
        if len(matrix.vdims) != 1:
            raise ValueError('Input matrix to gradient operation must have single value dimension.')
        matrix_dim = matrix.vdims[0]
        data = np.flipud(matrix.dimension_values(matrix_dim, flat=False))
        r, c = data.shape
        if matrix_dim.cyclic and None in matrix_dim.range:
            raise Exception('Cyclic range must be specified to compute the gradient of cyclic quantities')
        cyclic_range = None if not matrix_dim.cyclic else np.diff(matrix_dim.range)
        if cyclic_range is not None:
            data = data - matrix_dim.range[0]
        dx = np.diff(data, 1, axis=1)[0:r - 1, 0:c - 1]
        dy = np.diff(data, 1, axis=0)[0:r - 1, 0:c - 1]
        if cyclic_range is not None:
            dx = dx % cyclic_range
            dy = dy % cyclic_range
            dx_negatives = dx - cyclic_range
            dy_negatives = dy - cyclic_range
            dx = np.where(np.abs(dx_negatives) < dx, dx_negatives, dx)
            dy = np.where(np.abs(dy_negatives) < dy, dy_negatives, dy)
        return Image(np.sqrt(dx * dx + dy * dy), bounds=matrix.bounds, group=self.p.group)