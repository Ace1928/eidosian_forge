from ipywidgets import Widget, Color
from traitlets import Unicode, List, Enum, Float, Bool, Type, Tuple
import numpy as np
from .traits import Date
from ._version import __frontend_version__
@register_scale('bqplot.OrdinalScale')
class OrdinalScale(Scale):
    """An ordinal scale.

    A mapping from a discrete set of values to a numerical range.

    Attributes
    ----------
    domain: list (default: [])
        The discrete values mapped by the ordinal scale
    rtype: string (class-level attribute)
        This attribute should not be modified by the user.
        The range type of a linear scale is numerical.
    dtype: type (class-level attribute)
        the associated data type / domain type
    """
    rtype = 'Number'
    dtype = np.str_
    domain = List().tag(sync=True)
    _view_name = Unicode('OrdinalScale').tag(sync=True)
    _model_name = Unicode('OrdinalScaleModel').tag(sync=True)