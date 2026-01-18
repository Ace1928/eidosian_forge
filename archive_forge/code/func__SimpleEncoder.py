import struct
from cloudsdk.google.protobuf.internal import wire_format
def _SimpleEncoder(wire_type, encode_value, compute_value_size):
    """Return a constructor for an encoder for fields of a particular type.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      encode_value:  A function which encodes an individual value, e.g.
        _EncodeVarint().
      compute_value_size:  A function which computes the size of an individual
        value, e.g. _VarintSize().
  """

    def SpecificEncoder(field_number, is_repeated, is_packed):
        if is_packed:
            tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
            local_EncodeVarint = _EncodeVarint

            def EncodePackedField(write, value, deterministic):
                write(tag_bytes)
                size = 0
                for element in value:
                    size += compute_value_size(element)
                local_EncodeVarint(write, size, deterministic)
                for element in value:
                    encode_value(write, element, deterministic)
            return EncodePackedField
        elif is_repeated:
            tag_bytes = TagBytes(field_number, wire_type)

            def EncodeRepeatedField(write, value, deterministic):
                for element in value:
                    write(tag_bytes)
                    encode_value(write, element, deterministic)
            return EncodeRepeatedField
        else:
            tag_bytes = TagBytes(field_number, wire_type)

            def EncodeField(write, value, deterministic):
                write(tag_bytes)
                return encode_value(write, value, deterministic)
            return EncodeField
    return SpecificEncoder