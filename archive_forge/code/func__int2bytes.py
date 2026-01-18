from __future__ import absolute_import
import binascii
from struct import pack
from rsa._compat import byte, is_integer
from rsa import common, machine_size
def _int2bytes(number, block_size=None):
    """Converts a number to a string of bytes.

    Usage::

        >>> _int2bytes(123456789)
        b'\\x07[\\xcd\\x15'
        >>> bytes2int(_int2bytes(123456789))
        123456789

        >>> _int2bytes(123456789, 6)
        b'\\x00\\x00\\x07[\\xcd\\x15'
        >>> bytes2int(_int2bytes(123456789, 128))
        123456789

        >>> _int2bytes(123456789, 3)
        Traceback (most recent call last):
        ...
        OverflowError: Needed 4 bytes for number, but block size is 3

    @param number: the number to convert
    @param block_size: the number of bytes to output. If the number encoded to
        bytes is less than this, the block will be zero-padded. When not given,
        the returned block is not padded.

    @throws OverflowError when block_size is given and the number takes up more
        bytes than fit into the block.
    """
    if not is_integer(number):
        raise TypeError("You must pass an integer for 'number', not %s" % number.__class__)
    if number < 0:
        raise ValueError('Negative numbers cannot be used: %i' % number)
    if number == 0:
        needed_bytes = 1
        raw_bytes = [b'\x00']
    else:
        needed_bytes = common.byte_size(number)
        raw_bytes = []
    if block_size and block_size > 0:
        if needed_bytes > block_size:
            raise OverflowError('Needed %i bytes for number, but block size is %i' % (needed_bytes, block_size))
    while number > 0:
        raw_bytes.insert(0, byte(number & 255))
        number >>= 8
    if block_size and block_size > 0:
        padding = (block_size - needed_bytes) * b'\x00'
    else:
        padding = b''
    return padding + b''.join(raw_bytes)