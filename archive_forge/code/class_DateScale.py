from ipywidgets import Widget, Color
from traitlets import Unicode, List, Enum, Float, Bool, Type, Tuple
import numpy as np
from .traits import Date
from ._version import __frontend_version__
@register_scale('bqplot.DateScale')
class DateScale(Scale):
    """A date scale, with customizable formatting.

    An affine mapping from dates to a numerical range.

    Attributes
    ----------
    min: Date or None (default: None)
        if not None, min is the minimal value of the domain
    max: Date (default: None)
        if not None, max is the maximal value of the domain
    domain_class: type (default: Date)
         traitlet type used to validate values in of the domain of the scale.
    rtype: string (class-level attribute)
        This attribute should not be modified by the user.
        The range type of a linear scale is numerical.
    dtype: type (class-level attribute)
        the associated data type / domain type
    """
    rtype = 'Number'
    dtype = np.datetime64
    domain_class = Type(Date)
    min = Date(default_value=None, allow_none=True).tag(sync=True)
    max = Date(default_value=None, allow_none=True).tag(sync=True)
    _view_name = Unicode('DateScale').tag(sync=True)
    _model_name = Unicode('DateScaleModel').tag(sync=True)