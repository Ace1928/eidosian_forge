from ipywidgets import Widget, Color
from traitlets import Unicode, List, Enum, Float, Bool, Type, Tuple
import numpy as np
from .traits import Date
from ._version import __frontend_version__
@register_scale('bqplot.OrdinalColorScale')
class OrdinalColorScale(ColorScale):
    """An ordinal color scale.

    A mapping from a discrete set of values to colors.

    Attributes
    ----------
    domain: list (default: [])
        The discrete values mapped by the ordinal scales.
    rtype: string (class-level attribute)
        This attribute should not be modified by the user.
        The range type of a color scale is 'color'.
    dtype: type (class-level attribute)
        the associated data type / domain type
    """
    rtype = 'Color'
    dtype = np.str_
    domain = List().tag(sync=True)
    _view_name = Unicode('OrdinalColorScale').tag(sync=True)
    _model_name = Unicode('OrdinalColorScaleModel').tag(sync=True)