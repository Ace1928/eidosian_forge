from .api import Request, Response
from .types import (
class CreatePartitionsRequest_v1(Request):
    API_KEY = 37
    API_VERSION = 1
    SCHEMA = CreatePartitionsRequest_v0.SCHEMA
    RESPONSE_TYPE = CreatePartitionsResponse_v1