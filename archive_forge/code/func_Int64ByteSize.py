import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def Int64ByteSize(field_number, int64):
    return UInt64ByteSize(field_number, 18446744073709551615 & int64)