import os
import tempfile
import unittest
import onnx
class _OnnxTestTextualSerializer(onnx.serialization.ProtoSerializer):
    """Serialize and deserialize the ONNX textual representation."""
    supported_format = 'onnxtext'
    file_extensions = frozenset({'.onnxtext'})

    def serialize_proto(self, proto) -> bytes:
        text = onnx.printer.to_text(proto)
        return text.encode('utf-8')

    def deserialize_proto(self, serialized: bytes, proto):
        text = serialized.decode('utf-8')
        if isinstance(proto, onnx.ModelProto):
            return onnx.parser.parse_model(text)
        if isinstance(proto, onnx.GraphProto):
            return onnx.parser.parse_graph(text)
        if isinstance(proto, onnx.FunctionProto):
            return onnx.parser.parse_function(text)
        if isinstance(proto, onnx.NodeProto):
            return onnx.parser.parse_node(text)
        raise ValueError(f'Unsupported proto type: {type(proto)}')