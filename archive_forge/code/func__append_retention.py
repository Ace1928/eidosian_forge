from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_retention(params: List[str], retention: Optional[int]):
    """Append RETENTION property to params."""
    if retention is not None:
        params.extend(['RETENTION', retention])