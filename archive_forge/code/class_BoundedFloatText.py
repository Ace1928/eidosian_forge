from traitlets import (
from .widget_description import DescriptionWidget
from .trait_types import InstanceDict, NumberFormat
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from .widget_int import ProgressStyle, SliderStyle
@register
class BoundedFloatText(_BoundedFloat):
    """ Displays a float value within a textbox. Value must be within the range specified.

    For a textbox in which the value doesn't need to be within a specific range, use FloatText.

    Parameters
    ----------
    value : float
        value displayed
    min : float
        minimal value of the range of possible values displayed
    max : float
        maximal value of the range of possible values displayed
    step : float
        step of the increment (if None, any step is allowed)
    description : str
        description displayed next to the textbox
    """
    _view_name = Unicode('FloatTextView').tag(sync=True)
    _model_name = Unicode('BoundedFloatTextModel').tag(sync=True)
    disabled = Bool(False, help='Enable or disable user changes').tag(sync=True)
    continuous_update = Bool(False, help='Update the value as the user types. If False, update on submission, e.g., pressing Enter or navigating away.').tag(sync=True)
    step = CFloat(None, allow_none=True, help='Minimum step to increment the value').tag(sync=True)