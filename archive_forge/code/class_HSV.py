import colorsys
from copy import deepcopy
from operator import itemgetter
import numpy as np
import param
from ..core import Dataset, Dimension, Element2D, Overlay, config, util
from ..core.boundingregion import BoundingBox, BoundingRegion
from ..core.data import ImageInterface
from ..core.data.interface import DataError
from ..core.dimension import dimension_name
from ..core.sheetcoords import SheetCoordinateSystem, Slice
from .chart import Curve
from .geom import Selection2DExpr
from .graphs import TriMesh
from .tabular import Table
from .util import categorical_aggregate2d, compute_slice_bounds
class HSV(RGB):
    """
    HSV represents a regularly spaced 2D grid of an underlying
    continuous space of HSV (hue, saturation and value) color space
    values. The definition of the grid closely matches the semantics
    of an Image or RGB element and in the simplest case the grid may
    be specified as a NxMx3 or NxMx4 array of values along with a
    bounds, but it may also be defined through explicit and regularly
    spaced x/y-coordinate arrays. The two most basic supported
    constructors of an HSV element therefore include:

        HSV((X, Y, H, S, V))

    where X is a 1D array of shape M, Y is a 1D array of shape N and
    H/S/V are 2D array of shape NxM, or equivalently:

        HSV(Z, bounds=(x0, y0, x1, y1))

    where Z is a 3D array of stacked H/S/V arrays with shape NxMx3/4
    and the bounds define the (left, bottom, top, right) edges of the
    four corners of the grid. Other gridded formats which support
    declaring of explicit x/y-coordinate arrays such as xarray are
    also supported.

    Note that the interpretation of the orientation changes depending
    on whether bounds or explicit coordinates are used.
    """
    group = param.String(default='HSV', constant=True)
    alpha_dimension = param.ClassSelector(default=Dimension('A', range=(0, 1)), class_=Dimension, instantiate=False, doc='\n        The alpha dimension definition to add the value dimensions if\n        an alpha channel is supplied.')
    vdims = param.List(default=[Dimension('H', range=(0, 1), cyclic=True), Dimension('S', range=(0, 1)), Dimension('V', range=(0, 1))], bounds=(3, 4), doc='\n        The dimension description of the data held in the array.\n\n        If an alpha channel is supplied, the defined alpha_dimension\n        is automatically appended to this list.')
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    @property
    def rgb(self):
        """
        Conversion from HSV to RGB.
        """
        coords = tuple((self.dimension_values(d, expanded=False) for d in self.kdims))
        data = [self.dimension_values(d, flat=False) for d in self.vdims]
        hsv = self.hsv_to_rgb(*data[:3])
        if len(self.vdims) == 4:
            hsv += (data[3],)
        params = util.get_param_values(self)
        del params['vdims']
        return RGB(coords + hsv, bounds=self.bounds, xdensity=self.xdensity, ydensity=self.ydensity, **params)