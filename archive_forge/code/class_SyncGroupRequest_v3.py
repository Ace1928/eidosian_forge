from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class SyncGroupRequest_v3(Request):
    API_KEY = 14
    API_VERSION = 3
    RESPONSE_TYPE = SyncGroupResponse_v3
    SCHEMA = Schema(('group', String('utf-8')), ('generation_id', Int32), ('member_id', String('utf-8')), ('group_instance_id', String('utf-8')), ('group_assignment', Array(('member_id', String('utf-8')), ('member_metadata', Bytes))))