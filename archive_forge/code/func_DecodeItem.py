import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def DecodeItem(buffer, pos, end, message, field_dict):
    """Decode serialized message set to its value and new position.

    Args:
      buffer: memoryview of the serialized bytes.
      pos: int, position in the memory view to start at.
      end: int, end position of serialized data
      message: Message object to store unknown fields in
      field_dict: Map[Descriptor, Any] to store decoded values in.

    Returns:
      int, new position in serialized data.
    """
    message_set_item_start = pos
    type_id = -1
    message_start = -1
    message_end = -1
    while 1:
        tag_bytes, pos = local_ReadTag(buffer, pos)
        if tag_bytes == type_id_tag_bytes:
            type_id, pos = local_DecodeVarint(buffer, pos)
        elif tag_bytes == message_tag_bytes:
            size, message_start = local_DecodeVarint(buffer, pos)
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
    extension = message.Extensions._FindExtensionByNumber(type_id)
    if extension is not None:
        value = field_dict.get(extension)
        if value is None:
            message_type = extension.message_type
            if not hasattr(message_type, '_concrete_class'):
                message._FACTORY.GetPrototype(message_type)
            value = field_dict.setdefault(extension, message_type._concrete_class())
        if value._InternalParse(buffer, message_start, message_end) != message_end:
            raise _DecodeError('Unexpected end-group tag.')
    else:
        if not message._unknown_fields:
            message._unknown_fields = []
        message._unknown_fields.append((MESSAGE_SET_ITEM_TAG, buffer[message_set_item_start:pos].tobytes()))
        if message._unknown_field_set is None:
            message._unknown_field_set = containers.UnknownFieldSet()
        message._unknown_field_set._add(type_id, wire_format.WIRETYPE_LENGTH_DELIMITED, buffer[message_start:message_end].tobytes())
    return pos