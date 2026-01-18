import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def StringDecoder(field_number, is_repeated, is_packed, key, new_default, clear_if_default=False):
    """Returns a decoder for a string field."""
    local_DecodeVarint = _DecodeVarint

    def _ConvertToUnicode(memview):
        """Convert byte to unicode."""
        byte_str = memview.tobytes()
        try:
            value = str(byte_str, 'utf-8')
        except UnicodeDecodeError as e:
            e.reason = '%s in field: %s' % (e, key.full_name)
            raise
        return value
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
                value.append(_ConvertToUnicode(buffer[pos:new_pos]))
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
                field_dict[key] = _ConvertToUnicode(buffer[pos:new_pos])
            return new_pos
        return DecodeField