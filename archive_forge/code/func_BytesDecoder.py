import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def BytesDecoder(field_number, is_repeated, is_packed, key, new_default, clear_if_default=False):
    """Returns a decoder for a bytes field."""
    local_DecodeVarint = _DecodeVarint
    assert not is_packed
    if is_repeated:
        tag_bytes = encoder.TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
        tag_len = len(tag_bytes)

        def DecodeRepeatedField(buffer, pos, end, message, field_dict):
            value = field_dict.get(key)
            if value is None:
                value = field_dict.setdefault(key, new_default(message))
            while 1:
                size, pos = local_DecodeVarint(buffer, pos)
                new_pos = pos + size
                if new_pos > end:
                    raise _DecodeError('Truncated string.')
                value.append(buffer[pos:new_pos].tobytes())
                pos = new_pos + tag_len
                if buffer[new_pos:pos] != tag_bytes or new_pos == end:
                    return new_pos
        return DecodeRepeatedField
    else:

        def DecodeField(buffer, pos, end, message, field_dict):
            size, pos = local_DecodeVarint(buffer, pos)
            new_pos = pos + size
            if new_pos > end:
                raise _DecodeError('Truncated string.')
            if clear_if_default and (not size):
                field_dict.pop(key, None)
            else:
                field_dict[key] = buffer[pos:new_pos].tobytes()
            return new_pos
        return DecodeField