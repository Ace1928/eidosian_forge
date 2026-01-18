from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String, Bytes
class FetchResponse_v11(Response):
    API_KEY = 1
    API_VERSION = 11
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16), ('session_id', Int32), ('topics', Array(('topics', String('utf-8')), ('partitions', Array(('partition', Int32), ('error_code', Int16), ('highwater_offset', Int64), ('last_stable_offset', Int64), ('log_start_offset', Int64), ('aborted_transactions', Array(('producer_id', Int64), ('first_offset', Int64))), ('preferred_read_replica', Int32), ('message_set', Bytes))))))