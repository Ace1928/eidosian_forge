from __future__ import annotations
import logging # isort:skip
from json import JSONEncoder
from typing import Any
from ..settings import settings
from .serialization import Buffer, Serialized
class PayloadEncoder(JSONEncoder):

    def __init__(self, *, buffers: list[Buffer]=[], threshold: int=100, indent: int | None=None, separators: tuple[str, str] | None=None):
        super().__init__(sort_keys=False, allow_nan=False, indent=indent, separators=separators)
        self._buffers = {buf.id: buf for buf in buffers}
        self._threshold = threshold

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Buffer):
            if obj.id in self._buffers:
                return obj.ref
            else:
                return obj.to_base64()
        else:
            return super().default(obj)