import struct
from cloudsdk.google.protobuf.internal import wire_format
def PackedFieldSize(value):
    result = len(value) * value_size
    return result + local_VarintSize(result) + tag_size