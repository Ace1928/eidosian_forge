from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceRequest_v2(ProduceRequest):
    API_VERSION = 2
    RESPONSE_TYPE = ProduceResponse_v2
    SCHEMA = ProduceRequest_v1.SCHEMA