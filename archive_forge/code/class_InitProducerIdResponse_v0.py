from .api import Request, Response
from .types import Int16, Int32, Int64, Schema, String, Array, Boolean
class InitProducerIdResponse_v0(Response):
    API_KEY = 22
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16), ('producer_id', Int64), ('producer_epoch', Int16))