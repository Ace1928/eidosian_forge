from typing import (
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
def _add_prop_int(key: str, deft: Optional[int]) -> property:
    return property(lambda self: self._get_int(key, deft), lambda self, v: self._set_int(key, v), None, f'\n            Returns/Modify the status of {key}, Returns {deft} if not defined\n            ')