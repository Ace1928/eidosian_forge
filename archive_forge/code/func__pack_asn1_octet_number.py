import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def _pack_asn1_octet_number(num: int) -> bytes:
    """Packs an int number into an ASN.1 integer value that spans multiple octets."""
    num_octets = bytearray()
    while num:
        octet_value = num & 127
        if len(num_octets):
            octet_value |= 128
        num_octets.append(octet_value)
        num >>= 7
    num_octets.reverse()
    return num_octets