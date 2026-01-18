import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def EnumDecoder(field_number, is_repeated, is_packed, key, new_default, clear_if_default=False):
    """Returns a decoder for enum field."""
    enum_type = key.enum_type
    if is_packed:
        local_DecodeVarint = _DecodeVarint

        def DecodePackedField(buffer, pos, end, message, field_dict):
            """Decode serialized packed enum to its value and a new position.

      Args:
        buffer: memoryview of the serialized bytes.
        pos: int, position in the memory view to start at.
        end: int, end position of serialized data
        message: Message object to store unknown fields in
        field_dict: Map[Descriptor, Any] to store decoded values in.

      Returns:
        int, new position in serialized data.
      """
            value = field_dict.get(key)
            if value is None:
                value = field_dict.setdefault(key, new_default(message))
            endpoint, pos = local_DecodeVarint(buffer, pos)
            endpoint += pos
            if endpoint > end:
                raise _DecodeError('Truncated message.')
            while pos < endpoint:
                value_start_pos = pos
                element, pos = _DecodeSignedVarint32(buffer, pos)
                if element in enum_type.values_by_number:
                    value.append(element)
                else:
                    if not message._unknown_fields:
                        message._unknown_fields = []
                    tag_bytes = encoder.TagBytes(field_number, wire_format.WIRETYPE_VARINT)
                    message._unknown_fields.append((tag_bytes, buffer[value_start_pos:pos].tobytes()))
                    if message._unknown_field_set is None:
                        message._unknown_field_set = containers.UnknownFieldSet()
                    message._unknown_field_set._add(field_number, wire_format.WIRETYPE_VARINT, element)
            if pos > endpoint:
                if element in enum_type.values_by_number:
                    del value[-1]
                else:
                    del message._unknown_fields[-1]
                    del message._unknown_field_set._values[-1]
                raise _DecodeError('Packed element was truncated.')
            return pos
        return DecodePackedField
    elif is_repeated:
        tag_bytes = encoder.TagBytes(field_number, wire_format.WIRETYPE_VARINT)
        tag_len = len(tag_bytes)

        def DecodeRepeatedField(buffer, pos, end, message, field_dict):
            """Decode serialized repeated enum to its value and a new position.

      Args:
        buffer: memoryview of the serialized bytes.
        pos: int, position in the memory view to start at.
        end: int, end position of serialized data
        message: Message object to store unknown fields in
        field_dict: Map[Descriptor, Any] to store decoded values in.

      Returns:
        int, new position in serialized data.
      """
            value = field_dict.get(key)
            if value is None:
                value = field_dict.setdefault(key, new_default(message))
            while 1:
                element, new_pos = _DecodeSignedVarint32(buffer, pos)
                if element in enum_type.values_by_number:
                    value.append(element)
                else:
                    if not message._unknown_fields:
                        message._unknown_fields = []
                    message._unknown_fields.append((tag_bytes, buffer[pos:new_pos].tobytes()))
                    if message._unknown_field_set is None:
                        message._unknown_field_set = containers.UnknownFieldSet()
                    message._unknown_field_set._add(field_number, wire_format.WIRETYPE_VARINT, element)
                pos = new_pos + tag_len
                if buffer[new_pos:pos] != tag_bytes or new_pos >= end:
                    if new_pos > end:
                        raise _DecodeError('Truncated message.')
                    return new_pos
        return DecodeRepeatedField
    else:

        def DecodeField(buffer, pos, end, message, field_dict):
            """Decode serialized repeated enum to its value and a new position.

      Args:
        buffer: memoryview of the serialized bytes.
        pos: int, position in the memory view to start at.
        end: int, end position of serialized data
        message: Message object to store unknown fields in
        field_dict: Map[Descriptor, Any] to store decoded values in.

      Returns:
        int, new position in serialized data.
      """
            value_start_pos = pos
            enum_value, pos = _DecodeSignedVarint32(buffer, pos)
            if pos > end:
                raise _DecodeError('Truncated message.')
            if clear_if_default and (not enum_value):
                field_dict.pop(key, None)
                return pos
            if enum_value in enum_type.values_by_number:
                field_dict[key] = enum_value
            else:
                if not message._unknown_fields:
                    message._unknown_fields = []
                tag_bytes = encoder.TagBytes(field_number, wire_format.WIRETYPE_VARINT)
                message._unknown_fields.append((tag_bytes, buffer[value_start_pos:pos].tobytes()))
                if message._unknown_field_set is None:
                    message._unknown_field_set = containers.UnknownFieldSet()
                message._unknown_field_set._add(field_number, wire_format.WIRETYPE_VARINT, enum_value)
            return pos
        return DecodeField