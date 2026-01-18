from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceRequest_v5(ProduceRequest):
    """
    Same as v4. The version number is bumped since the v5 response includes an
    additional partition level field: the log_start_offset.
    """
    API_VERSION = 5
    RESPONSE_TYPE = ProduceResponse_v5
    SCHEMA = ProduceRequest_v4.SCHEMA