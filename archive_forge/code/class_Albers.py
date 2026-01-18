from ipywidgets import Widget, Color
from traitlets import Unicode, List, Enum, Float, Bool, Type, Tuple
import numpy as np
from .traits import Date
from ._version import __frontend_version__
@register_scale('bqplot.Albers')
class Albers(GeoScale):
    """A geographical scale which is an alias for a conic equal area projection.

    The Albers projection is a conic equal area map. It does not preserve scale
    or shape, though it is recommended for chloropleths since it preserves the
    relative areas of geographic features. Default values are US-centric.

    Attributes
    ----------
    scale_factor: float (default: 250)
        Specifies the scale value for the projection
    rotate: tuple (default: (96, 0))
        Degree of rotation in each axis.
    parallels: tuple (default: (29.5, 45.5))
        Sets the two parallels for the conic projection.
    center: tuple (default: (0, 60))
        Specifies the longitude and latitude where the map is centered.
    precision: float (default: 0.1)
        Specifies the threshold for the projections adaptive resampling to the
        specified value in pixels.
    rtype: (Number, Number) (class-level attribute)
        This attribute should not be modified. The range type of a geo
        scale is a tuple.
    dtype: type (class-level attribute)
        the associated data type / domain type
    """
    scale_factor = Float(250).tag(sync=True)
    rotate = Tuple((96, 0)).tag(sync=True)
    center = Tuple((0, 60)).tag(sync=True)
    parallels = Tuple((29.5, 45.5)).tag(sync=True)
    precision = Float(0.1).tag(sync=True)
    rtype = '(Number, Number)'
    dtype = np.number
    _view_name = Unicode('Albers').tag(sync=True)
    _model_name = Unicode('AlbersModel').tag(sync=True)