from .api import Request, Response
from .types import (
class DescribeGroupsResponse_v1(Response):
    API_KEY = 15
    API_VERSION = 1
    SCHEMA = Schema(('throttle_time_ms', Int32), ('groups', Array(('error_code', Int16), ('group', String('utf-8')), ('state', String('utf-8')), ('protocol_type', String('utf-8')), ('protocol', String('utf-8')), ('members', Array(('member_id', String('utf-8')), ('client_id', String('utf-8')), ('client_host', String('utf-8')), ('member_metadata', Bytes), ('member_assignment', Bytes))))))