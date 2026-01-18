import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def DecodePackedField(buffer, pos, end, message, field_dict):
    value = field_dict.get(key)
    if value is None:
        value = field_dict.setdefault(key, new_default(message))
    endpoint, pos = local_DecodeVarint(buffer, pos)
    endpoint += pos
    if endpoint > end:
        raise _DecodeError('Truncated message.')
    while pos < endpoint:
        element, pos = decode_value(buffer, pos)
        value.append(element)
    if pos > endpoint:
        del value[-1]
        raise _DecodeError('Packed element was truncated.')
    return pos