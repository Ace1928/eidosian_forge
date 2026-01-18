from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetCommitRequest_v2(Request):
    API_KEY = 8
    API_VERSION = 2
    RESPONSE_TYPE = OffsetCommitResponse_v2
    SCHEMA = Schema(('consumer_group', String('utf-8')), ('consumer_group_generation_id', Int32), ('consumer_id', String('utf-8')), ('retention_time', Int64), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('offset', Int64), ('metadata', String('utf-8')))))))
    DEFAULT_GENERATION_ID = -1
    DEFAULT_RETENTION_TIME = -1