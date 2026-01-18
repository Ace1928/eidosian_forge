from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
@property
def _max_wire_version(self) -> Optional[int]:
    return self.__max_wire_version