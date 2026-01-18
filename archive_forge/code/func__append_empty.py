from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_empty(params: List[str], empty: Optional[bool]):
    """Append EMPTY property to params."""
    if empty:
        params.append('EMPTY')