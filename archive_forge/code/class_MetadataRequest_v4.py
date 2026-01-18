from .api import Request, Response
from .types import Array, Boolean, Int16, Int32, Schema, String
class MetadataRequest_v4(Request):
    API_KEY = 3
    API_VERSION = 4
    RESPONSE_TYPE = MetadataResponse_v4
    SCHEMA = Schema(('topics', Array(String('utf-8'))), ('allow_auto_topic_creation', Boolean))
    ALL_TOPICS = -1
    NO_TOPICS = None