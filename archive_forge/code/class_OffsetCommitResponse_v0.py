from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetCommitResponse_v0(Response):
    API_KEY = 8
    API_VERSION = 0
    SCHEMA = Schema(('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('error_code', Int16))))))