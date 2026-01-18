from traitlets import (
from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget import register
from .trait_types import Color, NumberFormat
@register
class TagsInput(TagsInputBase):
    """
    List of string tags
    """
    _model_name = Unicode('TagsInputModel').tag(sync=True)
    _view_name = Unicode('TagsInputView').tag(sync=True)
    value = List(Unicode(), help='List of string tags').tag(sync=True)
    tag_style = CaselessStrEnum(values=['primary', 'success', 'info', 'warning', 'danger', ''], default_value='', help='Use a predefined styling for the tags.').tag(sync=True)