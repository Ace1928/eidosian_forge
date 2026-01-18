from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _index_list(key_or_list: _Hint, direction: Optional[Union[int, str]]=None) -> Sequence[tuple[str, Union[int, str, Mapping[str, Any]]]]:
    """Helper to generate a list of (key, direction) pairs.

    Takes such a list, or a single key, or a single key and direction.
    """
    if direction is not None:
        if not isinstance(key_or_list, str):
            raise TypeError('Expected a string and a direction')
        return [(key_or_list, direction)]
    else:
        if isinstance(key_or_list, str):
            return [(key_or_list, ASCENDING)]
        elif isinstance(key_or_list, abc.ItemsView):
            return list(key_or_list)
        elif isinstance(key_or_list, abc.Mapping):
            return list(key_or_list.items())
        elif not isinstance(key_or_list, (list, tuple)):
            raise TypeError('if no direction is specified, key_or_list must be an instance of list')
        values: list[tuple[str, int]] = []
        for item in key_or_list:
            if isinstance(item, str):
                item = (item, ASCENDING)
            values.append(item)
        return values