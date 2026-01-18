import struct
from cloudsdk.google.protobuf.internal import wire_format
def _FixedSizer(value_size):
    """Like _SimpleSizer except for a fixed-size field.  The input is the size
  of one value."""

    def SpecificSizer(field_number, is_repeated, is_packed):
        tag_size = _TagSize(field_number)
        if is_packed:
            local_VarintSize = _VarintSize

            def PackedFieldSize(value):
                result = len(value) * value_size
                return result + local_VarintSize(result) + tag_size
            return PackedFieldSize
        elif is_repeated:
            element_size = value_size + tag_size

            def RepeatedFieldSize(value):
                return len(value) * element_size
            return RepeatedFieldSize
        else:
            field_size = value_size + tag_size

            def FieldSize(value):
                return field_size
            return FieldSize
    return SpecificSizer