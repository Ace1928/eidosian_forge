from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetFetchRequest_v3(Request):
    API_KEY = 9
    API_VERSION = 3
    RESPONSE_TYPE = OffsetFetchResponse_v3
    SCHEMA = OffsetFetchRequest_v2.SCHEMA