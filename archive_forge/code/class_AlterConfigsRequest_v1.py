from .api import Request, Response
from .types import (
class AlterConfigsRequest_v1(Request):
    API_KEY = 33
    API_VERSION = 1
    RESPONSE_TYPE = AlterConfigsResponse_v1
    SCHEMA = AlterConfigsRequest_v0.SCHEMA