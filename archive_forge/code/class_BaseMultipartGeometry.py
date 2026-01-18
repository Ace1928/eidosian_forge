import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
class BaseMultipartGeometry(BaseGeometry):
    __slots__ = []

    @property
    def coords(self):
        raise NotImplementedError('Sub-geometries may have coordinate sequences, but multi-part geometries do not')

    @property
    def geoms(self):
        return GeometrySequence(self)

    def __bool__(self):
        return self.is_empty is False

    def __eq__(self, other):
        if not isinstance(other, BaseGeometry):
            return NotImplemented
        return type(other) == type(self) and len(self.geoms) == len(other.geoms) and all((a == b for a, b in zip(self.geoms, other.geoms)))

    def __hash__(self):
        return super().__hash__()

    def svg(self, scale_factor=1.0, color=None):
        """Returns a group of SVG elements for the multipart geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        color : str, optional
            Hex string for stroke or fill color. Default is to use "#66cc99"
            if geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if color is None:
            color = '#66cc99' if self.is_valid else '#ff3333'
        return '<g>' + ''.join((p.svg(scale_factor, color) for p in self.geoms)) + '</g>'