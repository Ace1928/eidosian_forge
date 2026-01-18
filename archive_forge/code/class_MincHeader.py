from __future__ import annotations
from numbers import Integral
import numpy as np
from .externals.netcdf import netcdf_file
from .fileslice import canonical_slicers
from .spatialimages import SpatialHeader, SpatialImage
class MincHeader(SpatialHeader):
    """Class to contain header for MINC formats"""
    data_layout = 'C'

    def data_to_fileobj(self, data, fileobj, rescale=True):
        """See Header class for an implementation we can't use"""
        raise NotImplementedError

    def data_from_fileobj(self, fileobj):
        """See Header class for an implementation we can't use"""
        raise NotImplementedError