from .api import Request, Response
from .types import (
class DeleteTopicsRequest_v3(Request):
    API_KEY = 20
    API_VERSION = 3
    RESPONSE_TYPE = DeleteTopicsResponse_v3
    SCHEMA = DeleteTopicsRequest_v0.SCHEMA