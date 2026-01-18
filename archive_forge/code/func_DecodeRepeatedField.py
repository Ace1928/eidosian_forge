import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def DecodeRepeatedField(buffer, pos, end, message, field_dict):
    value = field_dict.get(key)
    if value is None:
        value = field_dict.setdefault(key, new_default(message))
    while 1:
        element, new_pos = decode_value(buffer, pos)
        value.append(element)
        pos = new_pos + tag_len
        if buffer[new_pos:pos] != tag_bytes or new_pos >= end:
            if new_pos > end:
                raise _DecodeError('Truncated message.')
            return new_pos