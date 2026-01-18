import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def SFixed32ByteSize(field_number, sfixed32):
    return TagByteSize(field_number) + 4