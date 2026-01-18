import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def StringByteSize(field_number, string):
    return BytesByteSize(field_number, string.encode('utf-8'))