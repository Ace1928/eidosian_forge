from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Instance
from .trait_types import Color, InstanceDict, NumberFormat
from traitlets import (
def _int_doc(cls):
    """Add int docstring template to class init."""

    def __init__(self, value=None, **kwargs):
        if value is not None:
            kwargs['value'] = value
        super(cls, self).__init__(**kwargs)
    __init__.__doc__ = _int_doc_t
    cls.__init__ = __init__
    return cls