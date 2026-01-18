from __future__ import annotations
import warnings
import typing
from typing import Any, Collection, Optional, Protocol, TypeVar
import google.protobuf.json_format
import google.protobuf.message
import google.protobuf.text_format
import onnx
class ProtoSerializer(Protocol):
    """A serializer-deserializer to and from in-memory Protocol Buffers representations."""
    supported_format: str
    file_extensions: Collection[str]

    def serialize_proto(self, proto: _Proto) -> Any:
        """Serialize a in-memory proto to a serialized data type."""

    def deserialize_proto(self, serialized: Any, proto: _Proto) -> _Proto:
        """Parse a serialized data type into a in-memory proto."""