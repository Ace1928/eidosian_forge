from .api import Request, Response
from .types import (
class DescribeGroupsRequest_v1(Request):
    API_KEY = 15
    API_VERSION = 1
    RESPONSE_TYPE = DescribeGroupsResponse_v1
    SCHEMA = DescribeGroupsRequest_v0.SCHEMA