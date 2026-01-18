from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class HeartbeatRequest_v1(Request):
    API_KEY = 12
    API_VERSION = 1
    RESPONSE_TYPE = HeartbeatResponse_v1
    SCHEMA = HeartbeatRequest_v0.SCHEMA