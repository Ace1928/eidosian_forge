import struct
from cloudsdk.google.protobuf.internal import wire_format
def EncodeVarint(write, value, unused_deterministic=None):
    bits = value & 127
    value >>= 7
    while value:
        write(local_int2byte(128 | bits))
        bits = value & 127
        value >>= 7
    return write(local_int2byte(bits))