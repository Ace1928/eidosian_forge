from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class SyncGroupResponse_v0(Response):
    API_KEY = 14
    API_VERSION = 0
    SCHEMA = Schema(('error_code', Int16), ('member_assignment', Bytes))