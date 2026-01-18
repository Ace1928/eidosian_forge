import struct
from cloudsdk.google.protobuf.internal import wire_format
def MessageEncoder(field_number, is_repeated, is_packed):
    """Returns an encoder for a message field."""
    tag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
    local_EncodeVarint = _EncodeVarint
    assert not is_packed
    if is_repeated:

        def EncodeRepeatedField(write, value, deterministic):
            for element in value:
                write(tag)
                local_EncodeVarint(write, element.ByteSize(), deterministic)
                element._InternalSerialize(write, deterministic)
        return EncodeRepeatedField
    else:

        def EncodeField(write, value, deterministic):
            write(tag)
            local_EncodeVarint(write, value.ByteSize(), deterministic)
            return value._InternalSerialize(write, deterministic)
        return EncodeField