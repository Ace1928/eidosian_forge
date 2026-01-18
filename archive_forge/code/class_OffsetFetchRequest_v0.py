from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetFetchRequest_v0(Request):
    API_KEY = 9
    API_VERSION = 0
    RESPONSE_TYPE = OffsetFetchResponse_v0
    SCHEMA = Schema(('consumer_group', String('utf-8')), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(Int32)))))