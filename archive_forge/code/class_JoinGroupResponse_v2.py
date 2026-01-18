from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class JoinGroupResponse_v2(Response):
    API_KEY = 11
    API_VERSION = 2
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16), ('generation_id', Int32), ('group_protocol', String('utf-8')), ('leader_id', String('utf-8')), ('member_id', String('utf-8')), ('members', Array(('member_id', String('utf-8')), ('member_metadata', Bytes))))