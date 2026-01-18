import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def UInt64ByteSize(field_number, uint64):
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(uint64)