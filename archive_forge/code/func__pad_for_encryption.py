import hashlib
import os
from rsa._compat import range
from rsa import common, transform, core
def _pad_for_encryption(message, target_length):
    """Pads the message for encryption, returning the padded message.

    :return: 00 02 RANDOM_DATA 00 MESSAGE

    >>> block = _pad_for_encryption(b'hello', 16)
    >>> len(block)
    16
    >>> block[0:2]
    b'\\x00\\x02'
    >>> block[-6:]
    b'\\x00hello'

    """
    max_msglength = target_length - 11
    msglength = len(message)
    if msglength > max_msglength:
        raise OverflowError('%i bytes needed for message, but there is only space for %i' % (msglength, max_msglength))
    padding = b''
    padding_length = target_length - msglength - 3
    while len(padding) < padding_length:
        needed_bytes = padding_length - len(padding)
        new_padding = os.urandom(needed_bytes + 5)
        new_padding = new_padding.replace(b'\x00', b'')
        padding = padding + new_padding[:needed_bytes]
    assert len(padding) == padding_length
    return b''.join([b'\x00\x02', padding, b'\x00', message])