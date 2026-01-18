from typing import (
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
def _get_int(self, key: str, deft: Optional[NumberObject]) -> NumberObject:
    return self.get(key, deft)