from .api import Request, Response
from .types import (
class ApiVersionRequest_v0(Request):
    API_KEY = 18
    API_VERSION = 0
    RESPONSE_TYPE = ApiVersionResponse_v0
    SCHEMA = Schema()