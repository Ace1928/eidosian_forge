import struct
from cloudsdk.google.protobuf.internal import wire_format
def _ModifiedSizer(compute_value_size, modify_value):
    """Like SimpleSizer, but modify_value is invoked on each value before it is
  passed to compute_value_size.  modify_value is typically ZigZagEncode."""

    def SpecificSizer(field_number, is_repeated, is_packed):
        tag_size = _TagSize(field_number)
        if is_packed:
            local_VarintSize = _VarintSize

            def PackedFieldSize(value):
                result = 0
                for element in value:
                    result += compute_value_size(modify_value(element))
                return result + local_VarintSize(result) + tag_size
            return PackedFieldSize
        elif is_repeated:

            def RepeatedFieldSize(value):
                result = tag_size * len(value)
                for element in value:
                    result += compute_value_size(modify_value(element))
                return result
            return RepeatedFieldSize
        else:

            def FieldSize(value):
                return tag_size + compute_value_size(modify_value(value))
            return FieldSize
    return SpecificSizer