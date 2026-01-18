from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class SyncGroupResponse_v1(Response):
    API_KEY = 14
    API_VERSION = 1
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16), ('member_assignment', Bytes))