from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_bucket_timestamp(params: List[str], bucket_timestamp: Optional[str]):
    """Append BUCKET_TIMESTAMP property to params."""
    if bucket_timestamp is not None:
        params.extend(['BUCKETTIMESTAMP', bucket_timestamp])