from __future__ import annotations
import math
import typing as ty
from dataclasses import dataclass, replace
import numpy as np
from nibabel.casting import able_int_type
from nibabel.fileslice import strided_scalar
from nibabel.spatialimages import SpatialImage
def _homogeneous_coords(self):
    if self.homogeneous:
        return np.asanyarray(self.coordinates)
    ones = strided_scalar(shape=(self.coordinates.shape[0], 1), scalar=np.array(1, dtype=self.coordinates.dtype))
    return np.hstack((self.coordinates, ones))