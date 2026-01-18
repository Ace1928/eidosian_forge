import struct
from cloudsdk.google.protobuf.internal import wire_format
def EncodeNonFiniteOrRaise(write, value):
    if value == _POS_INF:
        write(b'\x00\x00\x00\x00\x00\x00\xf0\x7f')
    elif value == _NEG_INF:
        write(b'\x00\x00\x00\x00\x00\x00\xf0\xff')
    elif value != value:
        write(b'\x00\x00\x00\x00\x00\x00\xf8\x7f')
    else:
        raise