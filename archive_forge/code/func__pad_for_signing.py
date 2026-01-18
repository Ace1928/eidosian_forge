import hashlib
import os
from rsa._compat import range
from rsa import common, transform, core
def _pad_for_signing(message, target_length):
    """Pads the message for signing, returning the padded message.

    The padding is always a repetition of FF bytes.

    :return: 00 01 PADDING 00 MESSAGE

    >>> block = _pad_for_signing(b'hello', 16)
    >>> len(block)
    16
    >>> block[0:2]
    b'\\x00\\x01'
    >>> block[-6:]
    b'\\x00hello'
    >>> block[2:-6]
    b'\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\xff'

    """
    max_msglength = target_length - 11
    msglength = len(message)
    if msglength > max_msglength:
        raise OverflowError('%i bytes needed for message, but there is only space for %i' % (msglength, max_msglength))
    padding_length = target_length - msglength - 3
    return b''.join([b'\x00\x01', padding_length * b'\xff', b'\x00', message])