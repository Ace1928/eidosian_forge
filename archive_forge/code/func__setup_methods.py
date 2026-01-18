import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
def _setup_methods(cls):
    for method_info in cls.__info__.get_methods():
        setattr(cls, method_info.__name__, method_info)