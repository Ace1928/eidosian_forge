from .api import Request, Response
from .types import (
class SaslHandShakeRequest_v0(Request):
    API_KEY = 17
    API_VERSION = 0
    RESPONSE_TYPE = SaslHandShakeResponse_v0
    SCHEMA = Schema(('mechanism', String('utf-8')))