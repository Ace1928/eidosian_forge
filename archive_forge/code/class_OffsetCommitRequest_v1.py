from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetCommitRequest_v1(Request):
    API_KEY = 8
    API_VERSION = 1
    RESPONSE_TYPE = OffsetCommitResponse_v1
    SCHEMA = Schema(('consumer_group', String('utf-8')), ('consumer_group_generation_id', Int32), ('consumer_id', String('utf-8')), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('offset', Int64), ('timestamp', Int64), ('metadata', String('utf-8')))))))