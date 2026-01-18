import abc
from .struct import Struct
from .types import Int16, Int32, String, Schema, Array, TaggedFields
@abc.abstractproperty
def API_VERSION(self):
    """Integer of api request/response version"""
    pass