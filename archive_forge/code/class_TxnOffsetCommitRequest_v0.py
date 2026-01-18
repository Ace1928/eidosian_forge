from .api import Request, Response
from .types import Int16, Int32, Int64, Schema, String, Array, Boolean
class TxnOffsetCommitRequest_v0(Request):
    API_KEY = 28
    API_VERSION = 0
    RESPONSE_TYPE = TxnOffsetCommitResponse_v0
    SCHEMA = Schema(('transactional_id', String('utf-8')), ('group_id', String('utf-8')), ('producer_id', Int64), ('producer_epoch', Int16), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('offset', Int64), ('metadata', String('utf-8')))))))