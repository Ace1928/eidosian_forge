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
def _is_ascii_hex(b):
    if ord('0') <= b <= ord('9'):
        return True
    if ord('a') <= b <= ord('f'):
        return True
    return False