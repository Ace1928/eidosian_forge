import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def TagByteSize(field_number):
    """Returns the bytes required to serialize a tag with this field number."""
    return _VarUInt64ByteSizeNoTag(PackTag(field_number, 0))