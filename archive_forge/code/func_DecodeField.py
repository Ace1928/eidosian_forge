import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def DecodeField(buffer, pos, end, message, field_dict):
    new_value, pos = decode_value(buffer, pos)
    if pos > end:
        raise _DecodeError('Truncated message.')
    if clear_if_default and (not new_value):
        field_dict.pop(key, None)
    else:
        field_dict[key] = new_value
    return pos