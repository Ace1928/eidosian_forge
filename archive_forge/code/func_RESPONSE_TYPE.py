import abc
from .struct import Struct
from .types import Int16, Int32, String, Schema, Array, TaggedFields
@abc.abstractproperty
def RESPONSE_TYPE(self):
    """The Response class associated with the api request"""
    pass