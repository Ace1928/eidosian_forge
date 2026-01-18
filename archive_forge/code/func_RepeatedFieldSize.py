import struct
from cloudsdk.google.protobuf.internal import wire_format
def RepeatedFieldSize(value):
    return len(value) * element_size