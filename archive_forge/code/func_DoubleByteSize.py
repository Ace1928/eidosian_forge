import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def DoubleByteSize(field_number, double):
    return TagByteSize(field_number) + 8