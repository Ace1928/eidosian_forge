import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def _unpack_asn1_octet_number(b_data: bytes) -> typing.Tuple[int, int]:
    """Unpacks an ASN.1 INTEGER value that can span across multiple octets."""
    i = 0
    idx = 0
    while True:
        element = struct.unpack('B', b_data[idx:idx + 1])[0]
        idx += 1
        i = (i << 7) + (element & 127)
        if not element & 128:
            break
    return (i, idx)