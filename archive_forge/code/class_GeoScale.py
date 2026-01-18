from ipywidgets import Widget, Color
from traitlets import Unicode, List, Enum, Float, Bool, Type, Tuple
import numpy as np
from .traits import Date
from ._version import __frontend_version__
class GeoScale(Scale):
    """The base projection scale class for Map marks.

    The GeoScale represents a mapping between topographic data and a
    2d visual representation.
    """
    _view_name = Unicode('GeoScale').tag(sync=True)
    _model_name = Unicode('GeoScaleModel').tag(sync=True)