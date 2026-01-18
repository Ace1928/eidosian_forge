import struct
from cloudsdk.google.protobuf.internal import wire_format
def MapSizer(field_descriptor, is_message_map):
    """Returns a sizer for a map field."""
    message_type = field_descriptor.message_type
    message_sizer = MessageSizer(field_descriptor.number, False, False)

    def FieldSize(map_value):
        total = 0
        for key in map_value:
            value = map_value[key]
            entry_msg = message_type._concrete_class(key=key, value=value)
            total += message_sizer(entry_msg)
            if is_message_map:
                value.ByteSize()
        return total
    return FieldSize