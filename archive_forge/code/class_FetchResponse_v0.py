from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String, Bytes
class FetchResponse_v0(Response):
    API_KEY = 1
    API_VERSION = 0
    SCHEMA = Schema(('topics', Array(('topics', String('utf-8')), ('partitions', Array(('partition', Int32), ('error_code', Int16), ('highwater_offset', Int64), ('message_set', Bytes))))))