import struct
from cloudsdk.google.protobuf.internal import wire_format
def BytesSizer(field_number, is_repeated, is_packed):
    """Returns a sizer for a bytes field."""
    tag_size = _TagSize(field_number)
    local_VarintSize = _VarintSize
    local_len = len
    assert not is_packed
    if is_repeated:

        def RepeatedFieldSize(value):
            result = tag_size * len(value)
            for element in value:
                l = local_len(element)
                result += local_VarintSize(l) + l
            return result
        return RepeatedFieldSize
    else:

        def FieldSize(value):
            l = local_len(value)
            return tag_size + local_VarintSize(l) + l
        return FieldSize