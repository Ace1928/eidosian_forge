from traitlets import (
from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget import register
from .trait_types import Color, NumberFormat
@register
class FloatsInput(NumbersInputBase):
    """
    List of float tags
    """
    _model_name = Unicode('FloatsInputModel').tag(sync=True)
    _view_name = Unicode('FloatsInputView').tag(sync=True)
    value = List(CFloat(), help='List of float tags').tag(sync=True)
    format = NumberFormat('.1f').tag(sync=True)