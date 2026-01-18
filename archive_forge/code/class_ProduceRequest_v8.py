from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceRequest_v8(ProduceRequest):
    """
    V8 bumped up to add two new fields record_errors offset list and error_message to
    PartitionResponse (See KIP-467)
    """
    API_VERSION = 8
    RESPONSE_TYPE = ProduceResponse_v8
    SCHEMA = ProduceRequest_v7.SCHEMA