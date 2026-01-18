from .api import Request, Response
from .types import Array, Boolean, Int16, Int32, Schema, String
class MetadataResponse_v4(Response):
    API_KEY = 3
    API_VERSION = 4
    SCHEMA = MetadataResponse_v3.SCHEMA