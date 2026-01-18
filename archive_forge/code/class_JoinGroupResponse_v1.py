from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class JoinGroupResponse_v1(Response):
    API_KEY = 11
    API_VERSION = 1
    SCHEMA = JoinGroupResponse_v0.SCHEMA