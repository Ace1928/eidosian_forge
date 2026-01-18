from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _fields_list_to_dict(fields: Union[Mapping[str, Any], Iterable[str]], option_name: str) -> Mapping[str, Any]:
    """Takes a sequence of field names and returns a matching dictionary.

    ["a", "b"] becomes {"a": 1, "b": 1}

    and

    ["a.b.c", "d", "a.c"] becomes {"a.b.c": 1, "d": 1, "a.c": 1}
    """
    if isinstance(fields, abc.Mapping):
        return fields
    if isinstance(fields, (abc.Sequence, abc.Set)):
        if not all((isinstance(field, str) for field in fields)):
            raise TypeError(f'{option_name} must be a list of key names, each an instance of str')
        return dict.fromkeys(fields, 1)
    raise TypeError(f'{option_name} must be a mapping or list of key names')