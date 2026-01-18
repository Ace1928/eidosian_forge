from traitlets import (
from .widget_description import DescriptionWidget
from .trait_types import InstanceDict, NumberFormat
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from .widget_int import ProgressStyle, SliderStyle
class _Float(DescriptionWidget, ValueWidget, CoreWidget):
    value = CFloat(0.0, help='Float value').tag(sync=True)

    def __init__(self, value=None, **kwargs):
        if value is not None:
            kwargs['value'] = value
        super().__init__(**kwargs)