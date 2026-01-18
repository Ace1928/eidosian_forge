import collections.abc
from cloudsdk.google.protobuf import struct_pb2
from proto.marshal.collections import maps
from proto.marshal.collections import repeated
class ValueRule:
    """A rule to marshal between google.protobuf.Value and Python values."""

    def __init__(self, *, marshal):
        self._marshal = marshal

    def to_python(self, value, *, absent: bool=None):
        """Coerce the given value to the appropriate Python type.

        Note that both NullValue and absent fields return None.
        In order to disambiguate between these two options,
        use containment check,
        E.g.
        "value" in foo
        which is True for NullValue and False for an absent value.
        """
        kind = value.WhichOneof('kind')
        if kind == 'null_value' or absent:
            return None
        if kind == 'bool_value':
            return bool(value.bool_value)
        if kind == 'number_value':
            return float(value.number_value)
        if kind == 'string_value':
            return str(value.string_value)
        if kind == 'struct_value':
            return self._marshal.to_python(struct_pb2.Struct, value.struct_value, absent=False)
        if kind == 'list_value':
            return self._marshal.to_python(struct_pb2.ListValue, value.list_value, absent=False)
        raise ValueError('Unexpected kind: %s' % kind)

    def to_proto(self, value) -> struct_pb2.Value:
        """Return a protobuf Value object representing this value."""
        if isinstance(value, struct_pb2.Value):
            return value
        if value is None:
            return struct_pb2.Value(null_value=0)
        if isinstance(value, bool):
            return struct_pb2.Value(bool_value=value)
        if isinstance(value, (int, float)):
            return struct_pb2.Value(number_value=float(value))
        if isinstance(value, str):
            return struct_pb2.Value(string_value=value)
        if isinstance(value, collections.abc.Sequence):
            return struct_pb2.Value(list_value=self._marshal.to_proto(struct_pb2.ListValue, value))
        if isinstance(value, collections.abc.Mapping):
            return struct_pb2.Value(struct_value=self._marshal.to_proto(struct_pb2.Struct, value))
        raise ValueError('Unable to coerce value: %r' % value)