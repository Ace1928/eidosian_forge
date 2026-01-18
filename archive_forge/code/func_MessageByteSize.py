import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def MessageByteSize(field_number, message):
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(message.ByteSize()) + message.ByteSize()