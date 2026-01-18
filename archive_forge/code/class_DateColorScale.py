from ipywidgets import Widget, Color
from traitlets import Unicode, List, Enum, Float, Bool, Type, Tuple
import numpy as np
from .traits import Date
from ._version import __frontend_version__
@register_scale('bqplot.DateColorScale')
class DateColorScale(ColorScale):
    """A date color scale.

    A mapping from dates to a numerical domain.

    Attributes
    ----------
    min: Date or None (default: None)
        if not None, min is the minimal value of the domain
    max: Date or None (default: None)
        if not None, max is the maximal value of the domain
    mid: Date or None (default: None)
        if not None, mid is the value corresponding to the mid color.
    rtype: string (class-level attribute)
        This attribute should not be modified by the user.
        The range type of a color scale is 'Color'.
    dtype: type (class-level attribute)
        the associated data type / domain type
    """
    dtype = np.datetime64
    domain_class = Type(Date)
    min = Date(default_value=None, allow_none=True).tag(sync=True)
    mid = Date(default_value=None, allow_none=True).tag(sync=True)
    max = Date(default_value=None, allow_none=True).tag(sync=True)
    _view_name = Unicode('DateColorScale').tag(sync=True)
    _model_name = Unicode('DateColorScaleModel').tag(sync=True)