import struct
from cloudsdk.google.protobuf.internal import wire_format
def EncodeRepeatedField(write, value, unused_deterministic=None):
    for element in value:
        write(tag_bytes)
        try:
            write(local_struct_pack(format, element))
        except SystemError:
            EncodeNonFiniteOrRaise(write, element)