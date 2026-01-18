import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
def _setup_fields(cls):
    for field_info in cls.__info__.get_fields():
        name = field_info.get_name().replace('-', '_')
        setattr(cls, name, property(field_info.get_value, field_info.set_value))