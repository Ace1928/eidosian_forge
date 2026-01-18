from .api import Request, Response
from .types import (
class SaslHandShakeResponse_v0(Response):
    API_KEY = 17
    API_VERSION = 0
    SCHEMA = Schema(('error_code', Int16), ('enabled_mechanisms', Array(String('utf-8'))))