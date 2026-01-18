import struct
from cloudsdk.google.protobuf.internal import wire_format
def _VarintEncoder():
    """Return an encoder for a basic varint value (does not include tag)."""
    local_int2byte = struct.Struct('>B').pack

    def EncodeVarint(write, value, unused_deterministic=None):
        bits = value & 127
        value >>= 7
        while value:
            write(local_int2byte(128 | bits))
            bits = value & 127
            value >>= 7
        return write(local_int2byte(bits))
    return EncodeVarint