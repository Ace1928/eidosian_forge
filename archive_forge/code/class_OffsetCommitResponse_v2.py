from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetCommitResponse_v2(Response):
    API_KEY = 8
    API_VERSION = 2
    SCHEMA = OffsetCommitResponse_v1.SCHEMA