from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String, Bytes
class FetchResponse_v6(Response):
    """
    Same as FetchResponse_v5. The version number is bumped up to indicate that the
    client supports KafkaStorageException. The KafkaStorageException will be translated
    to NotLeaderForPartitionException in the response if version <= 5
    """
    API_KEY = 1
    API_VERSION = 6
    SCHEMA = FetchResponse_v5.SCHEMA