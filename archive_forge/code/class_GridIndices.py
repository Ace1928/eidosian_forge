from __future__ import annotations
import math
import typing as ty
from dataclasses import dataclass, replace
import numpy as np
from nibabel.casting import able_int_type
from nibabel.fileslice import strided_scalar
from nibabel.spatialimages import SpatialImage
class GridIndices:
    """Class for generating indices just-in-time"""
    __slots__ = ('gridshape', 'dtype', 'shape')
    ndim = 2

    def __init__(self, shape, dtype=None):
        self.gridshape = shape
        self.dtype = dtype or able_int_type(shape)
        self.shape = (math.prod(self.gridshape), len(self.gridshape))

    def __repr__(self):
        return f'<{self.__class__.__name__}{self.gridshape}>'

    def __array__(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
        axes = [np.arange(s, dtype=dtype) for s in self.gridshape]
        return np.reshape(np.meshgrid(*axes, copy=False, indexing='ij'), (len(axes), -1)).T