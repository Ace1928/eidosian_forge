from traitlets import (
from .widget_description import DescriptionWidget
from .trait_types import InstanceDict, NumberFormat
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from .widget_int import ProgressStyle, SliderStyle
@register
class FloatText(_Float):
    """ Displays a float value within a textbox. For a textbox in
    which the value must be within a specific range, use BoundedFloatText.

    Parameters
    ----------
    value : float
        value displayed
    step : float
        step of the increment (if None, any step is allowed)
    description : str
        description displayed next to the text box
    """
    _view_name = Unicode('FloatTextView').tag(sync=True)
    _model_name = Unicode('FloatTextModel').tag(sync=True)
    disabled = Bool(False, help='Enable or disable user changes').tag(sync=True)
    continuous_update = Bool(False, help='Update the value as the user types. If False, update on submission, e.g., pressing Enter or navigating away.').tag(sync=True)
    step = CFloat(None, allow_none=True, help='Minimum step to increment the value').tag(sync=True)