from .api import Request, Response
from .types import (
class CreateAclsRequest_v1(Request):
    API_KEY = 30
    API_VERSION = 1
    RESPONSE_TYPE = CreateAclsResponse_v1
    SCHEMA = Schema(('creations', Array(('resource_type', Int8), ('resource_name', String('utf-8')), ('resource_pattern_type', Int8), ('principal', String('utf-8')), ('host', String('utf-8')), ('operation', Int8), ('permission_type', Int8))))