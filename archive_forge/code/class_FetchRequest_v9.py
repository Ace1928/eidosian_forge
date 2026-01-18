from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String, Bytes
class FetchRequest_v9(Request):
    """
    adds the current leader epoch (see KIP-320)
    """
    API_KEY = 1
    API_VERSION = 9
    RESPONSE_TYPE = FetchResponse_v9
    SCHEMA = Schema(('replica_id', Int32), ('max_wait_time', Int32), ('min_bytes', Int32), ('max_bytes', Int32), ('isolation_level', Int8), ('session_id', Int32), ('session_epoch', Int32), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('current_leader_epoch', Int32), ('fetch_offset', Int64), ('log_start_offset', Int64), ('max_bytes', Int32))))), ('forgotten_topics_data', Array(('topic', String), ('partitions', Array(Int32)))))