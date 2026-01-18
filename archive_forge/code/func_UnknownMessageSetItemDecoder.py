import math
import struct
from google.protobuf.internal import containers
from google.protobuf.internal import encoder
from google.protobuf.internal import wire_format
from google.protobuf import message
def UnknownMessageSetItemDecoder():
    """Returns a decoder for a Unknown MessageSet item."""
    type_id_tag_bytes = encoder.TagBytes(2, wire_format.WIRETYPE_VARINT)
    message_tag_bytes = encoder.TagBytes(3, wire_format.WIRETYPE_LENGTH_DELIMITED)
    item_end_tag_bytes = encoder.TagBytes(1, wire_format.WIRETYPE_END_GROUP)

    def DecodeUnknownItem(buffer):
        pos = 0
        end = len(buffer)
        message_start = -1
        message_end = -1
        while 1:
            tag_bytes, pos = ReadTag(buffer, pos)
            if tag_bytes == type_id_tag_bytes:
                type_id, pos = _DecodeVarint(buffer, pos)
            elif tag_bytes == message_tag_bytes:
                size, message_start = _DecodeVarint(buffer, pos)
                pos = message_end = message_start + size
            elif tag_bytes == item_end_tag_bytes:
                break
            else:
                pos = SkipField(buffer, pos, end, tag_bytes)
                if pos == -1:
                    raise _DecodeError('Missing group end tag.')
        if pos > end:
            raise _DecodeError('Truncated message.')
        if type_id == -1:
            raise _DecodeError('MessageSet item missing type_id.')
        if message_start == -1:
            raise _DecodeError('MessageSet item missing message.')
        return (type_id, buffer[message_start:message_end].tobytes())
    return DecodeUnknownItem