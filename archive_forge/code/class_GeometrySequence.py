import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
class GeometrySequence:
    """
    Iterative access to members of a homogeneous multipart geometry.
    """
    _parent = None

    def __init__(self, parent):
        self._parent = parent

    def _get_geom_item(self, i):
        return shapely.get_geometry(self._parent, i)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self._get_geom_item(i)

    def __len__(self):
        return shapely.get_num_geometries(self._parent)

    def __getitem__(self, key):
        m = self.__len__()
        if isinstance(key, (int, np.integer)):
            if key + m < 0 or key >= m:
                raise IndexError('index out of range')
            if key < 0:
                i = m + key
            else:
                i = key
            return self._get_geom_item(i)
        elif isinstance(key, slice):
            res = []
            start, stop, stride = key.indices(m)
            for i in range(start, stop, stride):
                res.append(self._get_geom_item(i))
            return type(self._parent)(res or None)
        else:
            raise TypeError('key must be an index or slice')