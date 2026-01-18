from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceRequest_v7(ProduceRequest):
    """
    V7 bumped up to indicate ZStandard capability. (see KIP-110)
    """
    API_VERSION = 7
    RESPONSE_TYPE = ProduceResponse_v7
    SCHEMA = ProduceRequest_v6.SCHEMA