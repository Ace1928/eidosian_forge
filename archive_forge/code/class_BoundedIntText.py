from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Instance
from .trait_types import Color, InstanceDict, NumberFormat
from traitlets import (
@register
@_bounded_int_doc
class BoundedIntText(_BoundedInt):
    """Textbox widget that represents an integer bounded from above and below.
    """
    _view_name = Unicode('IntTextView').tag(sync=True)
    _model_name = Unicode('BoundedIntTextModel').tag(sync=True)
    disabled = Bool(False, help='Enable or disable user changes').tag(sync=True)
    continuous_update = Bool(False, help='Update the value as the user types. If False, update on submission, e.g., pressing Enter or navigating away.').tag(sync=True)
    step = CInt(1, help='Minimum step to increment the value').tag(sync=True)