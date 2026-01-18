from .widget_description import DescriptionStyle, DescriptionWidget
from .valuewidget import ValueWidget
from .widget import CallbackDispatcher, register, widget_serialization
from .widget_core import CoreWidget
from .trait_types import Color, InstanceDict, TypedTuple
from .utils import deprecation
from traitlets import Unicode, Bool, Int
@register
class LabelStyle(_StringStyle):
    """Label style widget."""
    _model_name = Unicode('LabelStyleModel').tag(sync=True)
    font_family = Unicode(None, allow_none=True, help='Label text font family.').tag(sync=True)
    font_style = Unicode(None, allow_none=True, help='Label text font style.').tag(sync=True)
    font_variant = Unicode(None, allow_none=True, help='Label text font variant.').tag(sync=True)
    font_weight = Unicode(None, allow_none=True, help='Label text font weight.').tag(sync=True)
    text_decoration = Unicode(None, allow_none=True, help='Label text decoration.').tag(sync=True)