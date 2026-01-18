from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_uncompressed(params: List[str], uncompressed: Optional[bool]):
    """Append UNCOMPRESSED tag to params."""
    if uncompressed:
        params.extend(['UNCOMPRESSED'])