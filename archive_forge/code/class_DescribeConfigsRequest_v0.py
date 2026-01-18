from .api import Request, Response
from .types import (
class DescribeConfigsRequest_v0(Request):
    API_KEY = 32
    API_VERSION = 0
    RESPONSE_TYPE = DescribeConfigsResponse_v0
    SCHEMA = Schema(('resources', Array(('resource_type', Int8), ('resource_name', String('utf-8')), ('config_names', Array(String('utf-8'))))))