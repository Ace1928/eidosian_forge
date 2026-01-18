from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String, Bytes
class FetchRequest_v8(Request):
    """
    bump used to indicate that on quota violation brokers send out responses before
    throttling.
    """
    API_KEY = 1
    API_VERSION = 8
    RESPONSE_TYPE = FetchResponse_v8
    SCHEMA = FetchRequest_v7.SCHEMA