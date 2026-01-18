from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetRequest_v4(Request):
    """
    Add current_leader_epoch to request
    """
    API_KEY = 2
    API_VERSION = 4
    RESPONSE_TYPE = OffsetResponse_v4
    SCHEMA = Schema(('replica_id', Int32), ('isolation_level', Int8), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('current_leader_epoch', Int64), ('timestamp', Int64))))))
    DEFAULTS = {'replica_id': -1}