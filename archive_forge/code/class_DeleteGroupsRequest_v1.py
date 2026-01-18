from .api import Request, Response
from .types import (
class DeleteGroupsRequest_v1(Request):
    API_KEY = 42
    API_VERSION = 1
    RESPONSE_TYPE = DeleteGroupsResponse_v1
    SCHEMA = DeleteGroupsRequest_v0.SCHEMA