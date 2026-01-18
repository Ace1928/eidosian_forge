from __future__ import annotations
import warnings
import typing
from typing import Any, Collection, Optional, Protocol, TypeVar
import google.protobuf.json_format
import google.protobuf.message
import google.protobuf.text_format
import onnx
class _TextualSerializer(ProtoSerializer):
    """Serialize and deserialize the ONNX textual representation."""
    supported_format = 'onnxtxt'
    file_extensions = frozenset({'.onnxtxt'})

    def serialize_proto(self, proto: _Proto) -> bytes:
        text = onnx.printer.to_text(proto)
        return text.encode(_ENCODING)

    def deserialize_proto(self, serialized: bytes | str, proto: _Proto) -> _Proto:
        warnings.warn('The onnxtxt format is experimental. Please report any errors to the ONNX GitHub repository.', stacklevel=2)
        if not isinstance(serialized, (bytes, str)):
            raise TypeError(f"Parameter 'serialized' must be bytes or str, but got type: {type(serialized)}")
        if isinstance(serialized, bytes):
            text = serialized.decode(_ENCODING)
        else:
            text = serialized
        if isinstance(proto, onnx.ModelProto):
            return onnx.parser.parse_model(text)
        if isinstance(proto, onnx.GraphProto):
            return onnx.parser.parse_graph(text)
        if isinstance(proto, onnx.FunctionProto):
            return onnx.parser.parse_function(text)
        if isinstance(proto, onnx.NodeProto):
            return onnx.parser.parse_node(text)
        raise ValueError(f'Unsupported proto type: {type(proto)}')