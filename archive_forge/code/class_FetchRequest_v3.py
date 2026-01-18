from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String, Bytes
class FetchRequest_v3(Request):
    API_KEY = 1
    API_VERSION = 3
    RESPONSE_TYPE = FetchResponse_v3
    SCHEMA = Schema(('replica_id', Int32), ('max_wait_time', Int32), ('min_bytes', Int32), ('max_bytes', Int32), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('offset', Int64), ('max_bytes', Int32))))))