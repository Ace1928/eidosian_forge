from __future__ import unicode_literals
import binascii
from collections import namedtuple
import six
import struct
import sys
from base64 import urlsafe_b64encode
from pymacaroons.utils import (
from pymacaroons.serializers.base_serializer import BaseSerializer
from pymacaroons.exceptions import MacaroonSerializationException
def _encode_uvarint(data, n):
    """ Encodes integer into variable-length format into data."""
    if n < 0:
        raise ValueError('only support positive integer')
    while True:
        this_byte = n & 127
        n >>= 7
        if n == 0:
            data.append(this_byte)
            break
        data.append(this_byte | 128)