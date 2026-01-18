from .api import Request, Response
from .types import Int16, Int32, Int64, Schema, String, Array, Boolean
class InitProducerIdRequest_v0(Request):
    API_KEY = 22
    API_VERSION = 0
    RESPONSE_TYPE = InitProducerIdResponse_v0
    SCHEMA = Schema(('transactional_id', String('utf-8')), ('transaction_timeout_ms', Int32))