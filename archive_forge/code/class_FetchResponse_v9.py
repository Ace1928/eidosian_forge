from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String, Bytes
class FetchResponse_v9(Response):
    API_KEY = 1
    API_VERSION = 9
    SCHEMA = FetchResponse_v7.SCHEMA