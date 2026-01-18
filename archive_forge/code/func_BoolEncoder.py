import struct
from cloudsdk.google.protobuf.internal import wire_format
def BoolEncoder(field_number, is_repeated, is_packed):
    """Returns an encoder for a boolean field."""
    false_byte = b'\x00'
    true_byte = b'\x01'
    if is_packed:
        tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
        local_EncodeVarint = _EncodeVarint

        def EncodePackedField(write, value, deterministic):
            write(tag_bytes)
            local_EncodeVarint(write, len(value), deterministic)
            for element in value:
                if element:
                    write(true_byte)
                else:
                    write(false_byte)
        return EncodePackedField
    elif is_repeated:
        tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_VARINT)

        def EncodeRepeatedField(write, value, unused_deterministic=None):
            for element in value:
                write(tag_bytes)
                if element:
                    write(true_byte)
                else:
                    write(false_byte)
        return EncodeRepeatedField
    else:
        tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_VARINT)

        def EncodeField(write, value, unused_deterministic=None):
            write(tag_bytes)
            if value:
                return write(true_byte)
            return write(false_byte)
        return EncodeField