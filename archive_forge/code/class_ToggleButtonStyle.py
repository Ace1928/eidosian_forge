from .widget_description import DescriptionStyle, DescriptionWidget
from .widget_core import CoreWidget
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .trait_types import Color, InstanceDict
from traitlets import Unicode, Bool, CaselessStrEnum
@register
class ToggleButtonStyle(DescriptionStyle, CoreWidget):
    """ToggleButton widget style."""
    _model_name = Unicode('ToggleButtonStyleModel').tag(sync=True)
    font_family = Unicode(None, allow_none=True, help='Toggle button text font family.').tag(sync=True)
    font_size = Unicode(None, allow_none=True, help='Toggle button text font size.').tag(sync=True)
    font_style = Unicode(None, allow_none=True, help='Toggle button text font style.').tag(sync=True)
    font_variant = Unicode(None, allow_none=True, help='Toggle button text font variant.').tag(sync=True)
    font_weight = Unicode(None, allow_none=True, help='Toggle button text font weight.').tag(sync=True)
    text_color = Color(None, allow_none=True, help='Toggle button text color').tag(sync=True)
    text_decoration = Unicode(None, allow_none=True, help='Toggle button text decoration.').tag(sync=True)