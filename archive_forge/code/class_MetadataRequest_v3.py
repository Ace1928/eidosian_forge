from .api import Request, Response
from .types import Array, Boolean, Int16, Int32, Schema, String
class MetadataRequest_v3(Request):
    API_KEY = 3
    API_VERSION = 3
    RESPONSE_TYPE = MetadataResponse_v3
    SCHEMA = MetadataRequest_v1.SCHEMA
    ALL_TOPICS = -1
    NO_TOPICS = None