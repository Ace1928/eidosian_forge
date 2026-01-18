import abc
from .struct import Struct
from .types import Int16, Int32, String, Schema, Array, TaggedFields
@abc.abstractproperty
def API_KEY(self):
    """Integer identifier for api request/response"""
    pass