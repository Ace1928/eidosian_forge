from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetFetchResponse_v3(Response):
    API_KEY = 9
    API_VERSION = 3
    SCHEMA = Schema(('throttle_time_ms', Int32), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('offset', Int64), ('metadata', String('utf-8')), ('error_code', Int16))))), ('error_code', Int16))