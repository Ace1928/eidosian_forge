from .api import Request, Response
from .types import (
class DeleteAclsRequest_v1(Request):
    API_KEY = 31
    API_VERSION = 1
    RESPONSE_TYPE = DeleteAclsResponse_v1
    SCHEMA = Schema(('filters', Array(('resource_type', Int8), ('resource_name', String('utf-8')), ('resource_pattern_type_filter', Int8), ('principal', String('utf-8')), ('host', String('utf-8')), ('operation', Int8), ('permission_type', Int8))))