from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Instance
from .trait_types import Color, InstanceDict, NumberFormat
from traitlets import (
@register
class Play(_BoundedInt):
    """Play/repeat buttons to step through values automatically, and optionally loop.
    """
    _view_name = Unicode('PlayView').tag(sync=True)
    _model_name = Unicode('PlayModel').tag(sync=True)
    playing = Bool(help='Whether the control is currently playing.').tag(sync=True)
    repeat = Bool(help='Whether the control will repeat in a continuous loop.').tag(sync=True)
    interval = CInt(100, help='The time between two animation steps (ms).').tag(sync=True)
    step = CInt(1, help='Increment step').tag(sync=True)
    disabled = Bool(False, help='Enable or disable user changes').tag(sync=True)
    show_repeat = Bool(True, help='Show the repeat toggle button in the widget.').tag(sync=True)