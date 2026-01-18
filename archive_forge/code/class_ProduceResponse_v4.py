from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceResponse_v4(Response):
    """
    The version number is bumped up to indicate that the client supports
    KafkaStorageException. The KafkaStorageException will be translated to
    NotLeaderForPartitionException in the response if version <= 3
    """
    API_KEY = 0
    API_VERSION = 4
    SCHEMA = ProduceResponse_v3.SCHEMA