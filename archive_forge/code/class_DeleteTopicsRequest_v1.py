from .api import Request, Response
from .types import (
class DeleteTopicsRequest_v1(Request):
    API_KEY = 20
    API_VERSION = 1
    RESPONSE_TYPE = DeleteTopicsResponse_v1
    SCHEMA = DeleteTopicsRequest_v0.SCHEMA