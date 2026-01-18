import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
class EmptyGeometry(BaseGeometry):

    def __new__(self):
        """Create an empty geometry."""
        warn("The 'EmptyGeometry()' constructor to create an empty geometry is deprecated, and will raise an error in the future. Use one of the geometry subclasses instead, for example 'GeometryCollection()'.", ShapelyDeprecationWarning, stacklevel=2)
        return shapely.from_wkt('GEOMETRYCOLLECTION EMPTY')