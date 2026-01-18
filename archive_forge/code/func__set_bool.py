from typing import (
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
def _set_bool(self, key: str, v: bool) -> None:
    self[NameObject(key)] = BooleanObject(v is True)