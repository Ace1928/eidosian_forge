from .api import Request, Response
from .types import (
class DescribeAclsRequest_v2(Request):
    """
    Enable flexible version
    """
    API_KEY = 29
    API_VERSION = 2
    RESPONSE_TYPE = DescribeAclsResponse_v2
    SCHEMA = DescribeAclsRequest_v1.SCHEMA