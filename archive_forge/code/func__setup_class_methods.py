import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
def _setup_class_methods(cls):
    info = cls.__info__
    class_struct = info.get_class_struct()
    if class_struct is None:
        return
    for method_info in class_struct.get_methods():
        name = method_info.__name__
        if not hasattr(cls, name):
            setattr(cls, name, classmethod(method_info))