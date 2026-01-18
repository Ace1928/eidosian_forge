from .api import Request, Response
from .types import Int16, Int32, Int64, Schema, String, Array, Boolean
class AddOffsetsToTxnRequest_v0(Request):
    API_KEY = 25
    API_VERSION = 0
    RESPONSE_TYPE = AddOffsetsToTxnResponse_v0
    SCHEMA = Schema(('transactional_id', String('utf-8')), ('producer_id', Int64), ('producer_epoch', Int16), ('group_id', String('utf-8')))