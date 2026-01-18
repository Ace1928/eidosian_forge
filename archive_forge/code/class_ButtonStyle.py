from .utils import deprecation
from .domwidget import DOMWidget
from .widget import CallbackDispatcher, register, widget_serialization
from .widget_core import CoreWidget
from .widget_style import Style
from .trait_types import Color, InstanceDict
from traitlets import Unicode, Bool, CaselessStrEnum, Instance, validate, default
@register
class ButtonStyle(Style, CoreWidget):
    """Button style widget."""
    _model_name = Unicode('ButtonStyleModel').tag(sync=True)
    button_color = Color(None, allow_none=True, help='Color of the button').tag(sync=True)
    font_family = Unicode(None, allow_none=True, help='Button text font family.').tag(sync=True)
    font_size = Unicode(None, allow_none=True, help='Button text font size.').tag(sync=True)
    font_style = Unicode(None, allow_none=True, help='Button text font style.').tag(sync=True)
    font_variant = Unicode(None, allow_none=True, help='Button text font variant.').tag(sync=True)
    font_weight = Unicode(None, allow_none=True, help='Button text font weight.').tag(sync=True)
    text_color = Unicode(None, allow_none=True, help='Button text color.').tag(sync=True)
    text_decoration = Unicode(None, allow_none=True, help='Button text decoration.').tag(sync=True)