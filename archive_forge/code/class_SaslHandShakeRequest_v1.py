from .api import Request, Response
from .types import (
class SaslHandShakeRequest_v1(Request):
    API_KEY = 17
    API_VERSION = 1
    RESPONSE_TYPE = SaslHandShakeResponse_v1
    SCHEMA = SaslHandShakeRequest_v0.SCHEMA