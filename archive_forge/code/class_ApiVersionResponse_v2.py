from .api import Request, Response
from .types import (
class ApiVersionResponse_v2(Response):
    API_KEY = 18
    API_VERSION = 2
    SCHEMA = ApiVersionResponse_v1.SCHEMA