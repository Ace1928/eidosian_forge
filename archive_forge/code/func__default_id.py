from ipywidgets import register, Widget, DOMWidget, widget_serialization
from traitlets import (
import uuid
from ._version import module_version
@default('_id')
def _default_id(self):
    return id_gen()