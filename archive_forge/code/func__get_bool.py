from typing import (
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
def _get_bool(self, key: str, deft: Optional[BooleanObject]) -> BooleanObject:
    return self.get(key, deft)