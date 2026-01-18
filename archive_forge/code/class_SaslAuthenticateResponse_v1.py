from .api import Request, Response
from .types import (
class SaslAuthenticateResponse_v1(Response):
    API_KEY = 36
    API_VERSION = 1
    SCHEMA = Schema(('error_code', Int16), ('error_message', String('utf-8')), ('sasl_auth_bytes', Bytes), ('session_lifetime_ms', Int64))