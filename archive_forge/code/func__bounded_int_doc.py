from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Instance
from .trait_types import Color, InstanceDict, NumberFormat
from traitlets import (
def _bounded_int_doc(cls):
    """Add bounded int docstring template to class init."""

    def __init__(self, value=None, min=None, max=None, step=None, **kwargs):
        if value is not None:
            kwargs['value'] = value
        if min is not None:
            kwargs['min'] = min
        if max is not None:
            kwargs['max'] = max
        if step is not None:
            kwargs['step'] = step
        super(cls, self).__init__(**kwargs)
    __init__.__doc__ = _bounded_int_doc_t
    cls.__init__ = __init__
    return cls