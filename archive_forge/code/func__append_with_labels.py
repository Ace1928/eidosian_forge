from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_with_labels(params: List[str], with_labels: Optional[bool], select_labels: Optional[List[str]]):
    """Append labels behavior to params."""
    if with_labels and select_labels:
        raise DataError('with_labels and select_labels cannot be provided together.')
    if with_labels:
        params.extend(['WITHLABELS'])
    if select_labels:
        params.extend(['SELECTED_LABELS', *select_labels])