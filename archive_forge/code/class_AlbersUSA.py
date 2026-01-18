from ipywidgets import Widget, Color
from traitlets import Unicode, List, Enum, Float, Bool, Type, Tuple
import numpy as np
from .traits import Date
from ._version import __frontend_version__
@register_scale('bqplot.AlbersUSA')
class AlbersUSA(GeoScale):
    """A composite projection of four Albers projections meant specifically for
    the United States.

    Attributes
    ----------
    scale_factor: float (default: 1200)
        Specifies the scale value for the projection
    translate: tuple (default: (600, 490))
    rtype: (Number, Number) (class-level attribute)
        This attribute should not be modified. The range type of a geo
        scale is a tuple.
    dtype: type (class-level attribute)
        the associated data type / domain type
    """
    scale_factor = Float(1200).tag(sync=True)
    translate = Tuple((600, 490)).tag(sync=True)
    rtype = '(Number, Number)'
    dtype = np.number
    _view_name = Unicode('AlbersUSA').tag(sync=True)
    _model_name = Unicode('AlbersUSAModel').tag(sync=True)