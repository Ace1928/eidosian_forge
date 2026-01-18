from .api import Request, Response
from .types import (
class SaslHandShakeResponse_v1(Response):
    API_KEY = 17
    API_VERSION = 1
    SCHEMA = SaslHandShakeResponse_v0.SCHEMA