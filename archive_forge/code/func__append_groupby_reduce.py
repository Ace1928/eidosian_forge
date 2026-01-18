from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_groupby_reduce(params: List[str], groupby: Optional[str], reduce: Optional[str]):
    """Append GROUPBY REDUCE property to params."""
    if groupby is not None and reduce is not None:
        params.extend(['GROUPBY', groupby, 'REDUCE', reduce.upper()])