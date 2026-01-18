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
def _exhaust_iterable(x):
    """Exhaust any non-mapping iterable into a tuple"""
    if isinstance(x, Iterable) and (not isinstance(x, Mapping)):
        return tuple(x)
    return x