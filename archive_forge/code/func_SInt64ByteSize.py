import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def SInt64ByteSize(field_number, int64):
    return UInt64ByteSize(field_number, ZigZagEncode(int64))