import struct
from cloudsdk.google.protobuf.internal import wire_format
def _StructPackEncoder(wire_type, format):
    """Return a constructor for an encoder for a fixed-width field.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      format:  The format string to pass to struct.pack().
  """
    value_size = struct.calcsize(format)

    def SpecificEncoder(field_number, is_repeated, is_packed):
        local_struct_pack = struct.pack
        if is_packed:
            tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
            local_EncodeVarint = _EncodeVarint

            def EncodePackedField(write, value, deterministic):
                write(tag_bytes)
                local_EncodeVarint(write, len(value) * value_size, deterministic)
                for element in value:
                    write(local_struct_pack(format, element))
            return EncodePackedField
        elif is_repeated:
            tag_bytes = TagBytes(field_number, wire_type)

            def EncodeRepeatedField(write, value, unused_deterministic=None):
                for element in value:
                    write(tag_bytes)
                    write(local_struct_pack(format, element))
            return EncodeRepeatedField
        else:
            tag_bytes = TagBytes(field_number, wire_type)

            def EncodeField(write, value, unused_deterministic=None):
                write(tag_bytes)
                return write(local_struct_pack(format, value))
            return EncodeField
    return SpecificEncoder