from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetRequest_v5(Request):
    API_KEY = 2
    API_VERSION = 5
    RESPONSE_TYPE = OffsetResponse_v5
    SCHEMA = OffsetRequest_v4.SCHEMA
    DEFAULTS = {'replica_id': -1}