from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class HeartbeatResponse_v0(Response):
    API_KEY = 12
    API_VERSION = 0
    SCHEMA = Schema(('error_code', Int16))