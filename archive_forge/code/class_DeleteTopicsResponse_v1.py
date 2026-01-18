from .api import Request, Response
from .types import (
class DeleteTopicsResponse_v1(Response):
    API_KEY = 20
    API_VERSION = 1
    SCHEMA = Schema(('throttle_time_ms', Int32), ('topic_error_codes', Array(('topic', String('utf-8')), ('error_code', Int16))))