from collections.abc import Iterable, Mapping
from itertools import chain
from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget_style import Style
from .trait_types import InstanceDict, TypedTuple
from .widget import register, widget_serialization
from .widget_int import SliderStyle
from .docutils import doc_subst
from traitlets import (Unicode, Bool, Int, Any, Dict, TraitError, CaselessStrEnum,
@register
class ToggleButtonsStyle(DescriptionStyle, CoreWidget):
    """Button style widget.

    Parameters
    ----------
    button_width: str
        The width of each button. This should be a valid CSS
        width, e.g. '10px' or '5em'.

    font_weight: str
        The text font weight of each button, This should be a valid CSS font
        weight unit, for example 'bold' or '600'
    """
    _model_name = Unicode('ToggleButtonsStyleModel').tag(sync=True)
    button_width = Unicode(help='The width of each button.').tag(sync=True)
    font_weight = Unicode(help='Text font weight of each button.').tag(sync=True)