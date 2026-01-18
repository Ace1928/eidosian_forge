import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
def _decodeLen(self, s):
    """Decode DER length octets from a file."""
    length = s.read_byte()
    if length > 127:
        encoded_length = s.read(length & 127)
        if bord(encoded_length[0]) == 0:
            raise ValueError('Invalid DER: length has leading zero')
        length = bytes_to_long(encoded_length)
        if length <= 127:
            raise ValueError('Invalid DER: length in long form but smaller than 128')
    return length