import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def FloatByteSize(field_number, flt):
    return TagByteSize(field_number) + 4