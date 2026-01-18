import logging
import numbers
from typing import Any, Callable, List, Optional, Tuple
from ray._private.dict import flatten_dict
from ray.air._internal.util import is_nan
from ray.air.config import MAX
from ray.train import CheckpointConfig
from ray.train._internal.session import _TrainingResult
from ray.train._internal.storage import _delete_fs_path
def _insert_into_sorted_list(list: List[Any], item: Any, key: Callable[[Any], Any]):
    """Insert an item into a sorted list with a custom key function.

    Examples:

        >>> list = []
        >>> _insert_into_sorted_list(list, {"a": 1, "b": 0}, lambda x: x["a"])
        >>> list
        [{'a': 1, 'b': 0}]
        >>> _insert_into_sorted_list(list, {"a": 3, "b": 1}, lambda x: x["a"])
        >>> list
        [{'a': 1, 'b': 0}, {'a': 3, 'b': 1}]
        >>> _insert_into_sorted_list(list, {"a": 4, "b": 2}, lambda x: x["a"])
        >>> list
        [{'a': 1, 'b': 0}, {'a': 3, 'b': 1}, {'a': 4, 'b': 2}]
        >>> _insert_into_sorted_list(list, {"a": 1, "b": 3}, lambda x: x["a"])
        >>> list
        [{'a': 1, 'b': 0}, {'a': 1, 'b': 3}, {'a': 3, 'b': 1}, {'a': 4, 'b': 2}]
    """
    i = 0
    while i < len(list):
        if key(list[i]) > key(item):
            break
        i += 1
    list.insert(i, item)