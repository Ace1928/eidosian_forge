import abc
from .struct import Struct
from .types import Int16, Int32, String, Schema, Array, TaggedFields
class ResponseHeaderV2(Struct):
    SCHEMA = Schema(('correlation_id', Int32), ('tags', TaggedFields))