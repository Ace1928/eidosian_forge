from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetResponse_v4(Response):
    """
    Add leader_epoch to response
    """
    API_KEY = 2
    API_VERSION = 4
    SCHEMA = Schema(('throttle_time_ms', Int32), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('error_code', Int16), ('timestamp', Int64), ('offset', Int64), ('leader_epoch', Int32))))))