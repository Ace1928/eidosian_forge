from .api import Request, Response
from .types import (
class DescribeGroupsRequest_v2(Request):
    API_KEY = 15
    API_VERSION = 2
    RESPONSE_TYPE = DescribeGroupsResponse_v2
    SCHEMA = DescribeGroupsRequest_v0.SCHEMA