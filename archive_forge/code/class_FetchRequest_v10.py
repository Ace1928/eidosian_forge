from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String, Bytes
class FetchRequest_v10(Request):
    """
    bumped up to indicate ZStandard capability. (see KIP-110)
    """
    API_KEY = 1
    API_VERSION = 10
    RESPONSE_TYPE = FetchResponse_v10
    SCHEMA = FetchRequest_v9.SCHEMA