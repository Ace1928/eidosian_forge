import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def SInt32ByteSize(field_number, int32):
    return UInt32ByteSize(field_number, ZigZagEncode(int32))