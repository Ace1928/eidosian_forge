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
def _coord2matrix(self, coord):
    return self.sheet2matrixidx(*coord)