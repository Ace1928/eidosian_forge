from .api import Request, Response
from .types import (
class CreateTopicsRequest_v2(Request):
    API_KEY = 19
    API_VERSION = 2
    RESPONSE_TYPE = CreateTopicsResponse_v2
    SCHEMA = CreateTopicsRequest_v1.SCHEMA