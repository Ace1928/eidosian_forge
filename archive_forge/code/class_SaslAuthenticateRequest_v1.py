from .api import Request, Response
from .types import (
class SaslAuthenticateRequest_v1(Request):
    API_KEY = 36
    API_VERSION = 1
    RESPONSE_TYPE = SaslAuthenticateResponse_v1
    SCHEMA = SaslAuthenticateRequest_v0.SCHEMA