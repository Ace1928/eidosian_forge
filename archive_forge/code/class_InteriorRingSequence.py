import numpy as np
import shapely
from shapely.algorithms.cga import is_ccw_impl, signed_area
from shapely.errors import TopologicalError
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
class InteriorRingSequence:
    _parent = None
    _ndim = None
    _index = 0
    _length = 0

    def __init__(self, parent):
        self._parent = parent
        self._ndim = parent._ndim

    def __iter__(self):
        self._index = 0
        self._length = self.__len__()
        return self

    def __next__(self):
        if self._index < self._length:
            ring = self._get_ring(self._index)
            self._index += 1
            return ring
        else:
            raise StopIteration

    def __len__(self):
        return shapely.get_num_interior_rings(self._parent)

    def __getitem__(self, key):
        m = self.__len__()
        if isinstance(key, int):
            if key + m < 0 or key >= m:
                raise IndexError('index out of range')
            if key < 0:
                i = m + key
            else:
                i = key
            return self._get_ring(i)
        elif isinstance(key, slice):
            res = []
            start, stop, stride = key.indices(m)
            for i in range(start, stop, stride):
                res.append(self._get_ring(i))
            return res
        else:
            raise TypeError('key must be an index or slice')

    def _get_ring(self, i):
        return shapely.get_interior_ring(self._parent, i)