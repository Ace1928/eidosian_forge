from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Instance
from .trait_types import Color, InstanceDict, NumberFormat
from traitlets import (
@register
class ProgressStyle(DescriptionStyle, CoreWidget):
    """Button style widget."""
    _model_name = Unicode('ProgressStyleModel').tag(sync=True)
    bar_color = Color(None, allow_none=True, help='Color of the progress bar.').tag(sync=True)