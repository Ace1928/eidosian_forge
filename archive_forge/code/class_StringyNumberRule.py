from proto.primitives import ProtoType
class StringyNumberRule:
    """A marshal between certain numeric types and strings

    This is a necessary hack to allow round trip conversion
    from messages to dicts back to messages.

    See https://github.com/protocolbuffers/protobuf/issues/2679
    and
    https://developers.google.com/protocol-buffers/docs/proto3#json
    for more details.
    """

    def to_python(self, value, *, absent: bool=None):
        return value

    def to_proto(self, value):
        if value is not None:
            return self._python_type(value)
        return None