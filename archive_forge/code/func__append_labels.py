from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_labels(params: List[str], labels: Optional[List[str]]):
    """Append LABELS property to params."""
    if labels:
        params.append('LABELS')
        for k, v in labels.items():
            params.extend([k, v])