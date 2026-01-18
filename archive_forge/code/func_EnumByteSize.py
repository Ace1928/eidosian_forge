import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def EnumByteSize(field_number, enum):
    return UInt32ByteSize(field_number, enum)