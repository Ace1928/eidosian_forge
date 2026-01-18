import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def GroupDecoder(field_number, is_repeated, is_packed, key, new_default):
    """Returns a decoder for a group field."""
    end_tag_bytes = encoder.TagBytes(field_number, wire_format.WIRETYPE_END_GROUP)
    end_tag_len = len(end_tag_bytes)
    assert not is_packed
    if is_repeated:
        tag_bytes = encoder.TagBytes(field_number, wire_format.WIRETYPE_START_GROUP)
        tag_len = len(tag_bytes)

        def DecodeRepeatedField(buffer, pos, end, message, field_dict):
            value = field_dict.get(key)
            if value is None:
                value = field_dict.setdefault(key, new_default(message))
            while 1:
                value = field_dict.get(key)
                if value is None:
                    value = field_dict.setdefault(key, new_default(message))
                pos = value.add()._InternalParse(buffer, pos, end)
                new_pos = pos + end_tag_len
                if buffer[pos:new_pos] != end_tag_bytes or new_pos > end:
                    raise _DecodeError('Missing group end tag.')
                pos = new_pos + tag_len
                if buffer[new_pos:pos] != tag_bytes or new_pos == end:
                    return new_pos
        return DecodeRepeatedField
    else:

        def DecodeField(buffer, pos, end, message, field_dict):
            value = field_dict.get(key)
            if value is None:
                value = field_dict.setdefault(key, new_default(message))
            pos = value._InternalParse(buffer, pos, end)
            new_pos = pos + end_tag_len
            if buffer[pos:new_pos] != end_tag_bytes or new_pos > end:
                raise _DecodeError('Missing group end tag.')
            return new_pos
        return DecodeField