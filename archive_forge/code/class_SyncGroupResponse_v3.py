from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class SyncGroupResponse_v3(Response):
    API_KEY = 14
    API_VERSION = 3
    SCHEMA = SyncGroupResponse_v1.SCHEMA