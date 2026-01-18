from .api import Request, Response
from .types import (
class AlterConfigsRequest_v0(Request):
    API_KEY = 33
    API_VERSION = 0
    RESPONSE_TYPE = AlterConfigsResponse_v0
    SCHEMA = Schema(('resources', Array(('resource_type', Int8), ('resource_name', String('utf-8')), ('config_entries', Array(('config_name', String('utf-8')), ('config_value', String('utf-8')))))), ('validate_only', Boolean))