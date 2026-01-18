from .api import Request, Response
from .types import (
class DeleteTopicsRequest_v2(Request):
    API_KEY = 20
    API_VERSION = 2
    RESPONSE_TYPE = DeleteTopicsResponse_v2
    SCHEMA = DeleteTopicsRequest_v0.SCHEMA