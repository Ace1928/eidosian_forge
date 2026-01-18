from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class JoinGroupRequest_v1(Request):
    API_KEY = 11
    API_VERSION = 1
    RESPONSE_TYPE = JoinGroupResponse_v1
    SCHEMA = Schema(('group', String('utf-8')), ('session_timeout', Int32), ('rebalance_timeout', Int32), ('member_id', String('utf-8')), ('protocol_type', String('utf-8')), ('group_protocols', Array(('protocol_name', String('utf-8')), ('protocol_metadata', Bytes))))
    UNKNOWN_MEMBER_ID = ''