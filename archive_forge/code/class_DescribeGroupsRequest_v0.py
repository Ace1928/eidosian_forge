from .api import Request, Response
from .types import (
class DescribeGroupsRequest_v0(Request):
    API_KEY = 15
    API_VERSION = 0
    RESPONSE_TYPE = DescribeGroupsResponse_v0
    SCHEMA = Schema(('groups', Array(String('utf-8'))))