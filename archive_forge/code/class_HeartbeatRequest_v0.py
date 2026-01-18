from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class HeartbeatRequest_v0(Request):
    API_KEY = 12
    API_VERSION = 0
    RESPONSE_TYPE = HeartbeatResponse_v0
    SCHEMA = Schema(('group', String('utf-8')), ('generation_id', Int32), ('member_id', String('utf-8')))