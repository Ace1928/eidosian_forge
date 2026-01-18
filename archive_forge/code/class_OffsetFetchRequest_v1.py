from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetFetchRequest_v1(Request):
    API_KEY = 9
    API_VERSION = 1
    RESPONSE_TYPE = OffsetFetchResponse_v1
    SCHEMA = OffsetFetchRequest_v0.SCHEMA