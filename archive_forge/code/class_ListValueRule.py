import collections.abc
from cloudsdk.google.protobuf import struct_pb2
from proto.marshal.collections import maps
from proto.marshal.collections import repeated
class ListValueRule:
    """A rule translating google.protobuf.ListValue and list-like objects."""

    def __init__(self, *, marshal):
        self._marshal = marshal

    def to_python(self, value, *, absent: bool=None):
        """Coerce the given value to a Python sequence."""
        return None if absent else repeated.RepeatedComposite(value.values, marshal=self._marshal)

    def to_proto(self, value) -> struct_pb2.ListValue:
        if isinstance(value, struct_pb2.ListValue):
            return value
        if isinstance(value, repeated.RepeatedComposite):
            return struct_pb2.ListValue(values=[v for v in value.pb])
        return struct_pb2.ListValue(values=[self._marshal.to_proto(struct_pb2.Value, v) for v in value])