from pyasn1.codec.der import decoder
from pyasn1_modules import pem
from pyasn1_modules.rfc2459 import Certificate
from pyasn1_modules.rfc5208 import PrivateKeyInfo
import rsa
import six
from oauth2client import _helpers
def _bit_list_to_bytes(bit_list):
    """Converts an iterable of 1's and 0's to bytes.

    Combines the list 8 at a time, treating each group of 8 bits
    as a single byte.
    """
    num_bits = len(bit_list)
    byte_vals = bytearray()
    for start in six.moves.xrange(0, num_bits, 8):
        curr_bits = bit_list[start:start + 8]
        char_val = sum((val * digit for val, digit in zip(_POW2, curr_bits)))
        byte_vals.append(char_val)
    return bytes(byte_vals)