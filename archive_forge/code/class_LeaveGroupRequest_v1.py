from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class LeaveGroupRequest_v1(Request):
    API_KEY = 13
    API_VERSION = 1
    RESPONSE_TYPE = LeaveGroupResponse_v1
    SCHEMA = LeaveGroupRequest_v0.SCHEMA