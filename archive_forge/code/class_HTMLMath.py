from .widget_description import DescriptionStyle, DescriptionWidget
from .valuewidget import ValueWidget
from .widget import CallbackDispatcher, register, widget_serialization
from .widget_core import CoreWidget
from .trait_types import Color, InstanceDict, TypedTuple
from .utils import deprecation
from traitlets import Unicode, Bool, Int
@register
class HTMLMath(_String):
    """Renders the string `value` as HTML, and render mathematics."""
    _view_name = Unicode('HTMLMathView').tag(sync=True)
    _model_name = Unicode('HTMLMathModel').tag(sync=True)
    style = InstanceDict(HTMLMathStyle).tag(sync=True, **widget_serialization)