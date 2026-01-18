from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
class OneDSelector(Selector):
    """One-dimensional selector interaction

    Base class for all selectors which select data in one dimension, i.e.,
    either the x or the y direction. The ``scale`` attribute should
    be provided.

    Attributes
    ----------
    scale: An instance of Scale
        This is the scale which is used for inversion from the pixels to data
        coordinates. This scale is used for setting the selected attribute for
        the selector.
    """
    scale = Instance(Scale, allow_none=True, default_value=None).tag(sync=True, dimension='x', **widget_serialization)
    _model_name = Unicode('OneDSelectorModel').tag(sync=True)