import struct
from cloudsdk.google.protobuf.internal import wire_format
def MessageSetItemSizer(field_number):
    """Returns a sizer for extensions of MessageSet.

  The message set message looks like this:
    message MessageSet {
      repeated group Item = 1 {
        required int32 type_id = 2;
        required string message = 3;
      }
    }
  """
    static_size = _TagSize(1) * 2 + _TagSize(2) + _VarintSize(field_number) + _TagSize(3)
    local_VarintSize = _VarintSize

    def FieldSize(value):
        l = value.ByteSize()
        return static_size + local_VarintSize(l) + l
    return FieldSize