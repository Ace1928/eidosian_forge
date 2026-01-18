from .api import Request, Response
from .types import (
class DescribeGroupsRequest_v3(Request):
    API_KEY = 15
    API_VERSION = 3
    RESPONSE_TYPE = DescribeGroupsResponse_v2
    SCHEMA = Schema(('groups', Array(String('utf-8'))), ('include_authorized_operations', Boolean))