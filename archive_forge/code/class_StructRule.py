import collections.abc
from cloudsdk.google.protobuf import struct_pb2
from proto.marshal.collections import maps
from proto.marshal.collections import repeated
class StructRule:
    """A rule translating google.protobuf.Struct and dict-like objects."""

    def __init__(self, *, marshal):
        self._marshal = marshal

    def to_python(self, value, *, absent: bool=None):
        """Coerce the given value to a Python mapping."""
        return None if absent else maps.MapComposite(value.fields, marshal=self._marshal)

    def to_proto(self, value) -> struct_pb2.Struct:
        if isinstance(value, struct_pb2.Struct):
            return value
        if isinstance(value, maps.MapComposite):
            return struct_pb2.Struct(fields={k: v for k, v in value.pb.items()})
        answer = struct_pb2.Struct(fields={k: self._marshal.to_proto(struct_pb2.Value, v) for k, v in value.items()})
        return answer