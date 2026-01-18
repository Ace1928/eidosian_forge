from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class MemberAssignment(Struct):
    SCHEMA = Schema(('version', Int16), ('assignment', Array(('topic', String('utf-8')), ('partitions', Array(Int32)))), ('user_data', Bytes))