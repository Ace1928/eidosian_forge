from .api import Request, Response
from .types import Array, Boolean, Int16, Int32, Schema, String
class MetadataRequest_v0(Request):
    API_KEY = 3
    API_VERSION = 0
    RESPONSE_TYPE = MetadataResponse_v0
    SCHEMA = Schema(('topics', Array(String('utf-8'))))
    ALL_TOPICS = None