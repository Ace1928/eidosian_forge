from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetRequest_v1(Request):
    API_KEY = 2
    API_VERSION = 1
    RESPONSE_TYPE = OffsetResponse_v1
    SCHEMA = Schema(('replica_id', Int32), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('timestamp', Int64))))))
    DEFAULTS = {'replica_id': -1}