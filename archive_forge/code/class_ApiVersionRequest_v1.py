from .api import Request, Response
from .types import (
class ApiVersionRequest_v1(Request):
    API_KEY = 18
    API_VERSION = 1
    RESPONSE_TYPE = ApiVersionResponse_v1
    SCHEMA = ApiVersionRequest_v0.SCHEMA