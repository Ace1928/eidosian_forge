from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetFetchRequest_v2(Request):
    API_KEY = 9
    API_VERSION = 2
    RESPONSE_TYPE = OffsetFetchResponse_v2
    SCHEMA = OffsetFetchRequest_v1.SCHEMA